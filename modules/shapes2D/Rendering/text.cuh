#pragma once

#include "framebuffer.cuh"

struct Font {
    uint32_t *texture;

    int tilesizeX = 26;
    int tilesizeY = 32;
    int numchars = 94;

    Font(uint32_t *texture) : texture(texture) {}

    float textureWidth() const { return numchars * tilesizeX; }
    float textureHeight() const { return tilesizeY; }

    float sample(char charcode, float u, float v) const {
        int tilepx = (charcode - 33) * tilesizeX;

        int sx = tilepx + tilesizeX * u;
        int sy = min(float(tilesizeY) * v, float(tilesizeY) - 1.0f);
        int sourceTexel = sx + sy * textureWidth();

        return texture[sourceTexel];
    }
};

class Cursor {
public:
    Font font;

    float fontsize;
    float posX, posY;
    float lineStartX;
    float3 textColor = {0.0f, 0.0f, 0.0f};

    Cursor(float fontsize, float x, float y, Font font)
        : font(font), fontsize(fontsize), posX(x), posY(y), lineStartX(x) {}
    float charSizeX() const { return fontsize / font.tilesizeY * font.tilesizeX; }
    float charSizeY() const { return fontsize; }
    void advance() { posX += charSizeX(); }
    void newline() {
        posY -= charSizeY() * 1.5f;
        posX = lineStartX;
    }
};

class TextRenderer {
private:
    Font font;

public:
    TextRenderer(Font font) : font(font) {}

    void drawText(const char *text, Cursor &cursor, Framebuffer framebuffer) {

        auto grid = cg::this_grid();

        int numchars = strlen(text);

        // one char after the other, utilizing 10k threads for each char haha
        for (int i = 0; i < numchars; i++) {

            int charcode = text[i];
            if (charcode == ' ') {
                cursor.advance();
                continue;
            }

            processRange(ceil(cursor.charSizeX()) * ceil(cursor.charSizeY()), [&](int index) {
                int l_x = index % int(ceil(cursor.charSizeX()));
                int l_y = index / int(ceil(cursor.charSizeX()));

                float u = float(l_x) / cursor.charSizeX();
                float v = 1.0f - float(l_y) / cursor.fontsize;

                float alpha = float4color(font.sample(charcode, u, v)).w;

                int t_x = l_x + cursor.posX;
                int t_y = l_y + cursor.posY;

                if (t_x < 0 || t_x >= framebuffer.width || t_y < 0 || t_y >= framebuffer.height) {
                    return;
                }

                int targetPixelIndex = t_x + t_y * framebuffer.width;

                // blend with current framebuffer value
                float4 color = make_float4(cursor.textColor, alpha);
                framebuffer.blend(targetPixelIndex, rgba8color(color));
            });

            grid.sync();

            cursor.advance();
        }
    }

    Cursor newCursor(float fontsize, float x, float y) { return Cursor(fontsize, x, y, font); }
};
