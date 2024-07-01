#include "common/helper_math.h"
#include "common/utils.cuh"

#include "HostDeviceInterface.h"
#include "World/Entities/entities.cuh"
#include "common/matrix_math.h"
#include "gui.cuh"

GUI::GUI(uint32_t width, uint32_t height, TextRenderer &textRenderer, SpriteSheet &sprites,
         mat4 viewProj)
    : textRenderer(textRenderer), sprites(sprites), viewProj(viewProj),
      moneyDisplay(width - 400, height - 100, 100, 30, sprites.moneyDisplay),
      populationDisplay(width - 400, height - 200, 100, 30, sprites.populationDisplay),
      timeDisplay(width - 400, height - 300, 100, 30, sprites.timeDisplay) {}

void GUI::render(Framebuffer framebuffer, Map *map, Entities *entities) {

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    renderInfoPanel(framebuffer, map, entities);

    GameState &gameState = *GameState::instance;
    char displayString[MAX_STRING_LENGTH + 1];

    unsigned int money = gameState.playerMoney;
    itos(money, displayString);
    renderDisplay(moneyDisplay, displayString, framebuffer);

    unsigned int population = gameState.population;
    itos(population, displayString);
    renderDisplay(populationDisplay, displayString, framebuffer);

    gameState.gameTime.formattedTime().clocktoString(displayString);
    renderDisplay(timeDisplay, displayString, framebuffer);
}

void GUI::renderInfoPanel(Framebuffer framebuffer, Map *map, Entities *entities) {
    auto grid = cg::this_grid();
    GameState &gameState = *GameState::instance;

    int id = gameState.buildingDisplay;
    if (id != -1) {
        float2 worldPos = map->getCellPosition(id);
        float2 screenPos = projectPosToScreenPos(make_float3(worldPos, 0.0f), viewProj,
                                                 framebuffer.width, framebuffer.height);
        GUIBox infoPanel(int(screenPos.x), int(screenPos.y), sprites.infoPanel.width,
                         sprites.infoPanel.height, sprites.infoPanel);

        infoPanel.sprite.draw(framebuffer, infoPanel.x, infoPanel.y, 3.0f);
        grid.sync();

        float line = infoPanel.y + 200.0f - 60.0f;

        switch (map->getTileId(id)) {
        case HOUSE: {
            textRenderer.drawText("House", infoPanel.x + 40.0f, line, 20, float3{0.0f, 0.0f, 0.0f},
                                  framebuffer);
            line -= 40.0f;

            int entityId = *map->houseTileData(id);
            if (entityId == -1) {
                textRenderer.drawText("No resident", infoPanel.x + 40.0f, line, 20,
                                      float3{0.0f, 0.0f, 0.0f}, framebuffer);
                line -= 30.0f;
            } else {
                auto &entity = entities->get(entityId);

                // resident money
                char displayString[30] = "Money: ";
                itos(entity.money, displayString + 7);
                textRenderer.drawText(displayString, infoPanel.x + 40.0f, line, 16,
                                      float3{0.0f, 0.0f, 0.0f}, framebuffer);
                line -= 30.0f;

                // resident job
                switch (map->getTileId(entity.workplaceId)) {
                case SHOP: {
                    strcpy(displayString, "Job: shop worker");
                    break;
                }
                case FACTORY: {
                    strcpy(displayString, "Job: factory worker");
                    break;
                }
                default:
                    strcpy(displayString, "Job: unemployed");
                    break;
                }
                textRenderer.drawText(displayString, infoPanel.x + 40.0f, line, 16,
                                      float3{0.0f, 0.0f, 0.0f}, framebuffer);
                line -= 30.0f;

                // resident current state
            }
            break;
        }
        case FACTORY: {
            textRenderer.drawText("Factory", infoPanel.x + 40.0f, line, 20,
                                  float3{0.0f, 0.0f, 0.0f}, framebuffer);
            line -= 40.0f;

            // open hours

            // worker count
            char displayString[30] = "Employees: ";
            itos(FACTORY_CAPACITY - *map->factoryTileData(id), displayString + 10);

            textRenderer.drawText(displayString, infoPanel.x + 40.0f, line, 16,
                                  float3{0.0f, 0.0f, 0.0f}, framebuffer);
            line -= 30.0f;

            break;
        }
        case SHOP: {
            textRenderer.drawText("Shop", infoPanel.x + 40.0f, line, 20, float3{0.0f, 0.0f, 0.0f},
                                  framebuffer);
            line -= 40.0f;

            // open hours

            // worker count
            char displayString[30] = "Employees: ";
            itos(SHOP_WORK_CAPACITY - map->shopWorkCapacity(id), displayString + 10);

            textRenderer.drawText(displayString, infoPanel.x + 40.0f, line, 16,
                                  float3{0.0f, 0.0f, 0.0f}, framebuffer);
            line -= 30.0f;

            break;
        }
        default:
            break;
        }
    }
}

void GUI::renderDisplay(GUIBox box, char *displayString, Framebuffer framebuffer) {
    auto grid = cg::this_grid();
    box.sprite.draw(framebuffer, box.x, box.y, 3.0f);
    grid.sync();
    textRenderer.drawText(displayString, box.x + 150.0, box.y + 30.0, 30, float3{0.0f, 0.0f, 0.0f},
                          framebuffer);
}