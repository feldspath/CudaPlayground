#pragma once

#include "HostDeviceInterface.h"
#include "World/map.cuh"

void findPath(
	int2 start, int2 end, Map map, GameData gamedata
	// uint32_t* costmap, uint32_t* distancefield, uint32_t* flowfield
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	

	int numCells = map.rows * map.cols;
	
	__shared__ uint32_t costmap[64 * 64];
	__shared__ uint32_t distancefield[64 * 64];
	__shared__ uint32_t flowfield[64 * 64];

	uint32_t endID = end.x + end.y * map.cols;

	uint64_t t_start = nanotime();

	// COMPUTE COSTMAP
	for(
		int cellID = block.thread_rank();
		cellID < numCells;
		cellID += block.size()
	){

		int cell_x = cellID % map.cols;
		int cell_y = cellID / map.cols;
		Cell& cell = map.cellsData[cellID];

		TileId type = cell.tileId;

		int cost = 1;
		if(type == TileId::STONE){
			cost = 10'000;
		}else if(type == TileId::WATER){
			cost = 10'000;
		}

		costmap[cellID] = cost;
	}

	block.sync();


	// COMPUTE DISTANCE FIELD

	// init distances
	for(
		int cellID = block.thread_rank();
		cellID < numCells;
		cellID += block.size()
	){
		distancefield[cellID] = cellID == endID ? 0 : 1'000'000'000;
	}

	block.sync();

	// uint64_t nanos = nanotime() - t_start;
	// float millies = double(nanos) / 1'000'000.0f;
	// if(grid.thread_rank() == 0) printf("millies: %f\n", millies);

	// now repeatedly compute distance towards goal, starting from goal
	// TODO: currently computed for all cells each iteration. Probably can be 
	// reduced to cells at radius from target with radius = iterations?
	for(int i = 0; i < map.rows; i++){
		for(
			int cellID = block.thread_rank();
			cellID < numCells;
			cellID += block.size()
		){
			// current cell is target, 
			// neighbors are sources that try to go to target

			int cell_x = cellID % map.cols;
			int cell_y = cellID / map.cols;

			int start_x = clamp(cell_x - 1, 0, map.cols - 1);
			int end_x   = clamp(cell_x + 1, 0, map.cols - 1);
			int start_y = clamp(cell_y - 1, 0, map.cols - 1);
			int end_y   = clamp(cell_y + 1, 0, map.cols - 1);

			int targetCost = costmap[cellID];
			int targetDistance = distancefield[cellID];


			for(int x = start_x; x <= end_x; x++)
			for(int y = start_y; y <= end_y; y++)
			{
				int sourceID = x + y * map.cols;

				if(sourceID == cellID) continue;

				float dx = cell_x - x;
				float dy = cell_y - y;
				float d = sqrt(dx * dx + dy * dy);

				uint32_t cost = targetDistance + targetCost;

				if(d > 1.0f) cost++;

				atomicMin(&distancefield[sourceID], cost);
			}
		}

		block.sync();
	}

	block.sync();

	// uint64_t nanos = nanotime() - t_start;
	// float millies = double(nanos) / 1'000'000.0f;
	// if(grid.thread_rank() == 0) printf("millies: %f\n", millies);

	// COMPUTE FLOW FIELD
	// for each cell, we compute the direction to the neighbor with the smallest distance
	for(
		int cellID = block.thread_rank();
		cellID < numCells;
		cellID += block.size()
	){
		int cell_x = cellID % map.cols;
		int cell_y = cellID / map.cols;

		int start_x = clamp(cell_x - 1, 0, map.cols - 1);
		int end_x   = clamp(cell_x + 1, 0, map.cols - 1);
		int start_y = clamp(cell_y - 1, 0, map.cols - 1);
		int end_y   = clamp(cell_y + 1, 0, map.cols - 1);

		int cheapestNeighborsDistance = 10'000'000'000;
		int cheapestNeighborIndex = 0;

		for(int x = start_x; x <= end_x; x++)
		for(int y = start_y; y <= end_y; y++)
		{
			int neighborID = x + y * map.cols;

			// if(neighborID == cellID) continue;

			int neighborsDistance = distancefield[neighborID];

			if(neighborsDistance < cheapestNeighborsDistance){
				cheapestNeighborsDistance = neighborsDistance;
				cheapestNeighborIndex = neighborID;
			}
		}

		// store index of neighbor we should move to
		flowfield[cellID] = cheapestNeighborIndex;
	}

	block.sync();


	// =============
	// DISPLAY PATHS
	// =============

	auto toCellID = [&](int x, int y){
		return x + y * map.cols;
	};

	auto fromCellID = [&](int id){
		int2 coord;
		coord.x = id % map.cols;
		coord.y = id / map.cols;

		return coord;
	};

	// if(false)
	if(threadIdx.x == 0){
		mat4 viewProj = gamedata.uniforms.proj * gamedata.uniforms.view;
		Entity& entity = gamedata.entities[blockIdx.x];
		

		// printf("[%d] %d, %d \n", blockIdx.x, end.x, end.y);

		int startID = toCellID(start.x, start.y);

		for(int i = 0; i < 50; i++){
			int nextID = flowfield[startID];
			float2 nextPos = make_float2(fromCellID(nextID)) + 0.5f;
			float2 currPos = make_float2(fromCellID(startID)) + 0.5f;

			if(startID == nextID) break;

			float2 screenPos = projectPosToScreenPos(
				make_float3(currPos.x, currPos.y, 0.0), viewProj, 
				gamedata.uniforms.width, gamedata.uniforms.height
			);

			float2 nextScreenPos = projectPosToScreenPos(
				make_float3(nextPos.x, nextPos.y, 0.0), viewProj, 
				gamedata.uniforms.width, gamedata.uniforms.height
			);

			Line line;
			line.start = screenPos;
			line.end = nextScreenPos;
			line.color = (blockIdx.x + 13) * 12345678;

			uint32_t lineIndex = atomicAdd(gamedata.numLines, 1);
			gamedata.lines[lineIndex] = line;

			startID = nextID;
		}

		


	}

}
