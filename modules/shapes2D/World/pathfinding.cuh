#pragma once

#include "HostDeviceInterface.h"
#include "World/map.cuh"

void findPath(int2 start, int2 end, Map map, GameData gamedata){

	auto grid = cg::this_grid();

	Allocator allocator(gamedata.buffer, 0);

	int numCells = map.rows * map.cols;
	
	uint32_t* costmap = allocator.alloc<uint32_t*>(4 * numCells);
	uint32_t* distancefield = allocator.alloc<uint32_t*>(4 * numCells);
	uint32_t* flowfield = allocator.alloc<uint32_t*>(4 * numCells);

	uint32_t endID = end.x + end.y * map.cols;

	uint64_t t_start = nanotime();

	// COMPUTE COSTMAP
	processRange(numCells, [&](int cellID){

		int cell_x = cellID % map.cols;
		int cell_y = cellID / map.cols;
		Cell& cell = map.cellsData[cellID];

		TileId type = cell.tileId;

		int cost = 1;
		if(type == TileId::STONE){
			cost = 10'000;
		}

		costmap[cellID] = cost;
	});

	grid.sync();

	// COMPUTE DISTANCE FIELD

	// init distances
	processRange(numCells, [&](int cellID){
		distancefield[cellID] = cellID == endID ? 0 : 1'000'000'000;
	});

	grid.sync();

	// uint64_t nanos = nanotime() - t_start;
	// float millies = double(nanos) / 1'000'000.0f;
	// if(grid.thread_rank() == 0) printf("millies: %f\n", millies);

	// now repeatedly compute distance towards goal, starting from goal
	// TODO: currently computed for all cells each iteration. Probably can be 
	// reduced to cells at radius from target with radius = iterations?
	for(int i = 0; i < map.rows; i++){
		processRange(numCells, [&](int cellID){
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
		});

		grid.sync();
	}

	grid.sync();

	// uint64_t nanos = nanotime() - t_start;
	// float millies = double(nanos) / 1'000'000.0f;
	// if(grid.thread_rank() == 0) printf("millies: %f\n", millies);

	// COMPUTE FLOW FIELD
	// for each cell, we compute the direction to the neighbor with the smallest distance
	processRange(numCells, [&](int cellID){
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
	});

	grid.sync();

	// uint64_t nanos = nanotime() - t_start;
	// float millies = double(nanos) / 1'000'000.0f;
	// if(grid.thread_rank() == 0) printf("millies: %f\n", millies);

	// DEBUG PRINT
	processRange(numCells, [&](int cellID){
		int cell_x = cellID % map.cols;
		int cell_y = cellID / map.cols;

		// int value = distancefield[cellID];
		int value = flowfield[cellID];

		mat4 viewProj = gamedata.uniforms.proj * gamedata.uniforms.view;
		float2 screenPos = projectPosToScreenPos(
			make_float3(cell_x, cell_y, 0.0), viewProj, 
			gamedata.uniforms.width, gamedata.uniforms.height
		);

		float2 nextScreenPos = projectPosToScreenPos(
			make_float3(cell_x + 1, cell_y + 1, 0.0), viewProj, 
			gamedata.uniforms.width, gamedata.uniforms.height
		);

		float2 diff_start = make_float2(cell_x - start.x, cell_y - start.y);
		float2 diff_end = make_float2(cell_x - end.x, cell_y - end.y);
		float l_start = length(diff_start);
		float l_end = length(diff_end);

		// if(cellID == 3420 - 5 * 64 + 10)
		if(l_start < 5.0 || l_end < 5.0f)
		{
			DbgLabel label;
			label.label[0] = 'A';
			label.size = 1;
			label.x = (screenPos.x + nextScreenPos.x) / 2.0f;
			label.y = (screenPos.y + nextScreenPos.y) / 2.0f;

			memset(&label.label[0], 0, 32);

			label.size = numberToString(value, &label.label[0]);

			uint32_t lblIndex = atomicAdd(gamedata.dbg_numLabels, 1);
			gamedata.dbg_labels[lblIndex] = label;
		}
	});



}

