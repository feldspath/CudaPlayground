#include "common/helper_math.h"
#include "common/utils.cuh"

#include "HostDeviceInterface.h"
#include "World/Entities/entities.cuh"
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

void GUI::render(const Map &map, Entities *entities) {

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    renderInfoPanel(map, entities);

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

void GUI::renderInfoPanel(const Map &map, Entities *entities) {
    auto grid = cg::this_grid();
    GameState &gameState = *GameState::instance;

    int id = gameState.buildingDisplay;
    if (id != -1) {
        float2 worldPos = map.getChunk(0).getCellPosition(id);
        float2 screenPos = projectPosToScreenPos(make_float3(worldPos, 0.0f), viewProj,
                                                 framebuffer.width, framebuffer.height);
        GUIBox infoPanel(int(screenPos.x), int(screenPos.y), sprites.infoPanel.width,
                         sprites.infoPanel.height, sprites.infoPanel);

        infoPanel.sprite.draw(framebuffer, infoPanel.x, infoPanel.y, 3.0f);
        grid.sync();

        Cursor cursor =
            textRenderer.newCursor(20.0f, infoPanel.x + 40.0f, infoPanel.y + 200.0f - 60.0f);
        char displayString[30];

        switch (map.getChunk(0).get(id).tileId) {
        case HOUSE: {
            const HouseCell &house = map.getChunk(0).getTyped<HouseCell>(id);
            textRenderer.drawText("House", cursor, framebuffer);
            cursor.newline();

            cursor.fontsize = 16.0;

            textRenderer.drawText("Resident count: ", cursor, framebuffer);
            itos(house.residentCount, displayString);
            textRenderer.drawText(displayString, cursor, framebuffer);
            textRenderer.drawText("/", cursor, framebuffer);
            itos(house.maxResidents(), displayString);
            textRenderer.drawText(displayString, cursor, framebuffer);
            cursor.newline();

            textRenderer.drawText("Level: ", cursor, framebuffer);
            itos(house.level + 1, displayString);
            textRenderer.drawText(displayString, cursor, framebuffer);
            cursor.newline();

            textRenderer.drawText("Wood count: ", cursor, framebuffer);
            itos(house.woodCount, displayString);
            textRenderer.drawText(displayString, cursor, framebuffer);
            textRenderer.drawText("/", cursor, framebuffer);
            itos(house.upgradeCost(), displayString);
            textRenderer.drawText(displayString, cursor, framebuffer);
            cursor.newline();
            break;
        }
        case FACTORY: {
            textRenderer.drawText("Wood Factory", cursor, framebuffer);
            cursor.newline();

            cursor.fontsize = 16.0;

            // worker count
            textRenderer.drawText("Employees: ", cursor, framebuffer);
            itos(FACTORY_CAPACITY - map.getChunk(0).getTyped<WorkplaceCell>(id).workplaceCapacity,
                 displayString);
            textRenderer.drawText(displayString, cursor, framebuffer);
            cursor.newline();

            textRenderer.drawText("Stock: ", cursor, framebuffer);
            itos(map.getChunk(0).getTyped<FactoryCell>(id).stockCount, displayString);
            textRenderer.drawText(displayString, cursor, framebuffer);
            cursor.newline();
            break;
        }
        case SHOP: {
            textRenderer.drawText("Shop", cursor, framebuffer);
            cursor.newline();

            cursor.fontsize = 16.0;

            // worker count
            textRenderer.drawText("Employees: ", cursor, framebuffer);
            itos(SHOP_WORK_CAPACITY - map.getChunk(0).getTyped<WorkplaceCell>(id).workplaceCapacity,
                 displayString);
            textRenderer.drawText(displayString, cursor, framebuffer);
            cursor.newline();

            textRenderer.drawText("Wood Stock: ", cursor, framebuffer);
            itos(map.getChunk(0).getTyped<ShopCell>(id).woodCount, displayString);
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