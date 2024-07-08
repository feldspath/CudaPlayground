#pragma once

#include <cooperative_groups.h>

#include "sprite.cuh"


struct ObjectSelectionSprite{
	float2 position;
	float2 size;
	Sprite sprite;
	const char* label;
	bool hovered = false;
	float depth = 1.0;
	float2 cellSize = {1, 1};
};

namespace ObjectSelection{

	


	ObjectSelectionSprite* createPanel(
		Allocator *allocator, uint32_t& out_numObjects, 
		int mouse_x, int mouse_y,
		GameData& gamedata
	){
		auto grid = cg::this_grid();

		auto& uniforms = gamedata.uniforms;

		const ObjectSelectionSprite OBJECT_SELECTION_SPRITES[3] = {
			{ // TOWN HALL
				.position = float2{0.0, 0.0},
				.size = float2{0.0, 0.0},
				.sprite = Sprite{
					.x = 0,
					.y = 0,
					.width = 256,
					.height = 256,
					.textureSize = {2048, 2048},
					.data = nullptr,
				},
				.label = "Town Hall",
				.cellSize = {6, 6},
			},
			{ // HOUSE
				.position = float2{0.0, 0.0},
				.size = float2{0.0, 0.0},
				.sprite = Sprite{
					.x = 256,
					.y = 0,
					.width = 256,
					.height = 256,
					.textureSize = {2048, 2048},
					.data = nullptr,
				},
				.label = "House",
				.cellSize = {3, 3},
			},
			{ // FORESTER
				.position = float2{0.0, 0.0},
				.size = float2{0.0, 0.0},
				.sprite = Sprite{
					.x = 512,
					.y = 0,
					.width = 256,
					.height = 256,
					.textureSize = {2048, 2048},
					.data = nullptr,
				},
				.label = "Forester",
				.cellSize = {2, 2},
			},
		};

		int numObjects = sizeof(OBJECT_SELECTION_SPRITES) / sizeof(OBJECT_SELECTION_SPRITES[0]);

		ObjectSelectionSprite* objects = allocator->alloc<ObjectSelectionSprite*>(sizeof(OBJECT_SELECTION_SPRITES));
		
		memcpy(objects, &OBJECT_SELECTION_SPRITES[0], sizeof(OBJECT_SELECTION_SPRITES));

		grid.sync();

		auto isHovered = [&](ObjectSelectionSprite& object){

            bool insideX = mouse_x > object.position.x && mouse_x < object.position.x + object.size.x;
            bool insideY = mouse_y > object.position.y && mouse_y < object.position.y + object.size.y;

            return insideX && insideY;
        };

        // update panel data
        if(grid.thread_rank() == 0){

            float spriteSize = 128;
            float panelSize = numObjects * spriteSize;
            float panelStart = uniforms.width / 2 - panelSize / 2;

            for(int i = 0; i < numObjects; i++){

                ObjectSelectionSprite object = objects[i];

                object.position.x = panelStart + i * spriteSize;
                object.position.y = 10;
                object.size = {spriteSize, spriteSize};
                object.sprite.data = gamedata.img_spritesheet_buildings;
				object.depth = 0.1;

                if(isHovered(object)){
                    object.position.x -= object.size.x * 0.4f;
                    object.position.y -= 1;
                    object.size = object.size * 1.8f;
					object.depth = 0.05;
                    object.hovered = true;
                }else{
                    object.hovered = false;
                }

                objects[i] = object;
            }
        }

		out_numObjects = numObjects;

		return objects;
	}

	void rasterize_blockwise(ObjectSelectionSprite object, Framebuffer framebuffer, bool black = false){

		auto block = cg::this_thread_block();

		float2 position = object.position;
		float2 size = object.size;

		int spriteWidth = size.x;
		int spriteHeight = size.y;
		int numPixels = spriteWidth * spriteHeight;

		
		for(
			int spritePixelID = block.thread_rank(); 
			spritePixelID < numPixels; 
			spritePixelID += block.size())
		{
			int spriteX = spritePixelID % spriteWidth;
			int spriteY = spritePixelID / spriteWidth;

			int2 pixelCoords = make_int2(position.x + spriteX, position.y + spriteY);

			if(pixelCoords.x < 0 || pixelCoords.x >= framebuffer.width) continue;
			if(pixelCoords.y < 0 || pixelCoords.y >= framebuffer.height) continue;

			int pixelID = pixelCoords.x + pixelCoords.y * framebuffer.width;
			pixelID = clamp(pixelID, 0, int(framebuffer.width * framebuffer.height) - 1);

			float u = float(spriteX) / spriteWidth;
			float v = float(spriteY) / spriteHeight;
			
			// float depth = position.y * 0.001;
			float depth = object.depth;
			uint32_t color = object.sprite.sample(u, 1.0f - v);
			uint8_t alpha = color >> 24;   

			if(black){
				color = black;
			}

			uint64_t udepth = *((uint32_t *)&depth);
			uint64_t pixel = (udepth << 32) | color;
			
			if(alpha > 0){
				atomicMin(&framebuffer.data[pixelID], pixel);
			}
		}

	}




};

