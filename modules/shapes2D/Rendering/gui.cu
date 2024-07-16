#include "common/helper_math.h"
#include "common/utils.cuh"

#include "HostDeviceInterface.h"
#include "common/matrix_math.h"
#include "gui.cuh"

GUI::GUI(Framebuffer &framebuffer, TextRenderer &textRenderer, SpriteSheet &sprites, mat4 viewProj)
    : textRenderer(textRenderer), framebuffer(framebuffer), sprites(sprites), viewProj(viewProj),
      moneyDisplay(framebuffer.width - 400, framebuffer.height - 100, 100, 30,
                   sprites.moneyDisplay),
      populationDisplay(framebuffer.width - 400, framebuffer.height - 200, 100, 30,
                        sprites.populationDisplay),
      timeDisplay(framebuffer.width - 400, framebuffer.height - 300, 100, 30, sprites.timeDisplay) {
}

void GUI::render(Map map, Entity *entities, uint32_t numEntities) {

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    renderInfoPanel(map, entities, numEntities);

    GameState &gameState = *GameState::instance;
    char displayString[MAX_STRING_LENGTH + 1];

    unsigned int money = gameState.playerMoney;
    itos(money, displayString);
    renderDisplay(moneyDisplay, displayString);

    unsigned int population = gameState.population;
    itos(population, displayString);
    renderDisplay(populationDisplay, displayString);

    gameState.gameTime.timeOfDay().clocktoString(displayString);
    renderDisplay(timeDisplay, displayString);
}

void GUI::renderInfoPanel(Map map, Entity *entities, uint32_t numEntities) {
    auto grid = cg::this_grid();
    GameState &gameState = *GameState::instance;

    int id = gameState.buildingDisplay;
    if (id != -1) {
        float2 worldPos = map.getCellPosition(id);
        float2 screenPos = projectPosToScreenPos(make_float3(worldPos, 0.0f), viewProj,
                                                 framebuffer.width, framebuffer.height);
        GUIBox infoPanel(int(screenPos.x), int(screenPos.y), sprites.infoPanel.width,
                         sprites.infoPanel.height, sprites.infoPanel);

        infoPanel.sprite.draw(framebuffer, infoPanel.x, infoPanel.y, 3.0f);
        grid.sync();

        Cursor cursor =
            textRenderer.newCursor(20.0f, infoPanel.x + 40.0f, infoPanel.y + 200.0f - 60.0f);
        char displayString[30];

        switch (map.getTileId(id)) {
        case HOUSE: {
            textRenderer.drawText("House", cursor, framebuffer);
            cursor.newline();

            int entityId = *map.houseTileData(id);
            if (entityId == -1) {
                textRenderer.drawText("No resident", cursor, framebuffer);
                cursor.newline();
            } else {
                Entity& entity = entities[entityId];
                cursor.fontsize = 16.0;

                // resident money
                textRenderer.drawText("Money: ", cursor, framebuffer);
                itos(entity.money, displayString);
                textRenderer.drawText(displayString, cursor, framebuffer);
                cursor.newline();

                // resident job
                switch (map.getTileId(entity.workplaceId)) {
                case SHOP: {
                    textRenderer.drawText("Job: shop worker", cursor, framebuffer);
                    break;
                }
                case FACTORY: {
                    textRenderer.drawText("Job: factory worker", cursor, framebuffer);
                    break;
                }
                default:
                    textRenderer.drawText("Job: unemployed", cursor, framebuffer);
                    break;
                }
                cursor.newline();

                // resident current state

                // resident happinness
                textRenderer.drawText("Happiness: ", cursor, framebuffer);
                float happiness_pct = int(float(entity.happiness) / 255.0f * 100.0f);
                itos(happiness_pct, displayString);
                textRenderer.drawText(displayString, cursor, framebuffer);
                cursor.newline();

                // rent cost
                textRenderer.drawText("Rent: ", cursor, framebuffer);
                itos(map.rentCost(entity.houseId), displayString);
                textRenderer.drawText(displayString, cursor, framebuffer);
                cursor.newline();
            }
            break;
        }
        case FACTORY: {
            textRenderer.drawText("Factory", cursor, framebuffer);
            cursor.newline();

            cursor.fontsize = 16.0;

            // open hours

            // worker count
            textRenderer.drawText("Employees: ", cursor, framebuffer);
            itos(FACTORY_CAPACITY - *map.factoryTileData(id), displayString);
            textRenderer.drawText(displayString, cursor, framebuffer);
            cursor.newline();
            break;
        }
        case SHOP: {
            textRenderer.drawText("Shop", cursor, framebuffer);
            cursor.newline();

            cursor.fontsize = 16.0;

            // open hours

            // worker count
            textRenderer.drawText("Employees: ", cursor, framebuffer);
            itos(SHOP_WORK_CAPACITY - map.shopWorkCapacity(id), displayString);
            textRenderer.drawText(displayString, cursor, framebuffer);
            cursor.newline();
            break;
        }
        default:
            break;
        }
    }
}

void GUI::renderDisplay(GUIBox box, char *displayString) {
    auto grid = cg::this_grid();
    box.sprite.draw(framebuffer, box.x, box.y, 3.0f);
    grid.sync();
    Cursor cursor = textRenderer.newCursor(30, box.x + 150.0, box.y + 30.0);
    textRenderer.drawText(displayString, cursor, framebuffer);
}