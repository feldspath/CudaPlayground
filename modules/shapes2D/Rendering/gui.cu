#include "common/utils.cuh"

#include "HostDeviceInterface.h"
#include "gui.cuh"

GUI::GUI(uint32_t width, uint32_t height, TextRenderer &textRenderer, SpriteSheet &sprites)
    : textRenderer(textRenderer), sprites(sprites),
      moneyDisplay(width - 400, height - 100, 100, 30, sprites.moneyDisplay),
      populationDisplay(width - 400, height - 200, 100, 30, sprites.populationDisplay),
      timeDisplay(width - 400, height - 300, 100, 30, sprites.timeDisplay) {}

void GUI::render(Framebuffer framebuffer) {
    char displayString[MAX_STRING_LENGTH + 1];

    unsigned int money = GameState::instance->playerMoney;
    itos(money, displayString);
    renderDisplay(moneyDisplay, displayString, framebuffer);

    unsigned int population = GameState::instance->population;
    itos(population, displayString);
    renderDisplay(populationDisplay, displayString, framebuffer);

    GameState::instance->gameTime.formattedTime().clocktoString(displayString);
    renderDisplay(timeDisplay, displayString, framebuffer);
}

void GUI::renderDisplay(GUIBox box, char *displayString, Framebuffer framebuffer) {
    auto grid = cg::this_grid();

    box.sprite.draw(framebuffer, box.x, box.y, 3.0f);

    grid.sync();

    textRenderer.drawText(displayString, box.x + 150.0, box.y + 30.0, 30, float3{0.0f, 0.0f, 0.0f},
                          framebuffer);
}