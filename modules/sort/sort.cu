
// Adaption of GPUSorting by Thomas Smith
// 
// https://github.com/b0nes164/GPUSorting/tree/main
// https://github.com/b0nes164/GPUSorting/blob/main/GPUSortingCUDA/DeviceRadixSort.cuh
// https://github.com/b0nes164/GPUSorting/blob/main/GPUSortingCUDA/DeviceRadixSort.cu
// Author of GPUSorting: Thomas Smith
// LICENSE: MIT


#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "../common/utils.cuh"

namespace cg = cooperative_groups;

constexpr int numElements = 64;
// constexpr int numElements = 5'000'000;
constexpr int maxValue = 1000;

constexpr bool TRACE = true;


void radix_sort(
	uint64_t* numbers, uint32_t count,
	uint32_t* globalCounters, uint32_t* globalCounters_blocks, 
	uint32_t* globalPrefixSum, uint32_t* globalPrefixSums_blocks,
	uint64_t* intermediateStorage,
	int pass
){


}


extern "C" __global__
void kernel(
	unsigned int* buffer,
	unsigned int* input
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	// allows allocating bytes from buffer
	Allocator allocator(buffer, 0);

	// allocate array of random values and initialize random number generators
	uint64_t* randomValues = allocator.alloc<uint64_t*>(numElements * sizeof(uint64_t));
	curandStateXORWOW_t thread_random_state;
	curand_init(grid.thread_rank(), 0, 0, &thread_random_state);


	int bins[10] = {0, 100, 200, 300, 400, 500, 600, 700, 800, 900};
	uint32_t* randomBinCounters = allocator.alloc<uint32_t*>(10 * sizeof(uint32_t));
	uint64_t* randomValues_0 = allocator.alloc<uint64_t*>(numElements * sizeof(uint64_t));

	if(grid.thread_rank() == 0){
		for(int i = 0; i < 10; i++){
			randomBinCounters[i] = 0;
		}
	}

	grid.sync();

	processRange(0, numElements, [&](int index){
		uint32_t randomValue = curand(&thread_random_state);
		uint32_t randomBinIndex = randomValue % 10;

		uint32_t counter = atomicAdd(&randomBinCounters[randomBinIndex], 1);

		if(counter < 20){
			randomValues_0[index] = bins[randomBinIndex];
		}else{
			randomValues_0[index] = 999;
		}
		
		
	});

	// store random values in array
	processRange(0, numElements, [&](int index){
		uint64_t randomValue = curand(&thread_random_state);
		uint64_t random_0_100 = randomValue % (maxValue + 1);
		randomValues[index] = random_0_100;
	});

	randomValues = randomValues_0;

	// globally sync all threads (wait until all numbers are generated)
	grid.sync();

	float average;
	uint64_t& sum = *allocator.alloc<uint64_t*>(8);
	{ // compute average
		if(grid.thread_rank() == 0){
			sum = 0;
		}
		grid.sync();

		// sum up all values
		processRange(0, numElements, [&](int index){
			atomicAdd(&sum, randomValues[index]);
		});

		grid.sync();
		
		average = double(sum) / double(numElements);
	}

	grid.sync();


	uint32_t* globalCounters          = allocator.alloc<uint32_t*>(4 * 65536); // Assumes max 16 bits are sorted at a time
	uint32_t* globalPrefixSums        = allocator.alloc<uint32_t*>(4 * 65536); // Assumes max 16 bits are sorted at a time
	uint32_t* globalPrefixSums_blocks = allocator.alloc<uint32_t*>(4 * 65536 * grid.num_blocks()); // Assumes max 16 bits are sorted at a time
	uint32_t* globalCounters_blocks   = allocator.alloc<uint32_t*>(4 * 65536 * grid.num_blocks()); // Assumes max 16 bits are sorted at a time



	uint64_t* intermediateStorage_0 = allocator.alloc<uint64_t*>(8 * numElements); 
	uint64_t* intermediateStorage_1 = allocator.alloc<uint64_t*>(8 * numElements); 

	uint64_t t_start = nanotime();
	grid.sync();
	for(int i = 0; i < 1; i++){
		
		radix_sort(randomValues, numElements, 
			globalCounters, globalCounters_blocks, 
			globalPrefixSums, globalPrefixSums_blocks, 
			intermediateStorage_0, 0);

		radix_sort(intermediateStorage_0, numElements, 
			globalCounters, globalCounters_blocks, 
			globalPrefixSums, globalPrefixSums_blocks, 
			intermediateStorage_1, 1);

	}

	grid.sync();

	if(grid.thread_rank() == 0){
		uint64_t nanos = nanotime() - t_start;
		double millies = double(nanos) / 1'000'000.0;

		printf("radix_sort duration: %.1f ms \n", millies);
	}
	
	// print stats and some of the random numbers
	// disable printing to see real kernel performance
	// if(false)
	if(grid.thread_rank() == 0){

		printf("created ");
		printNumber(numElements);
		printf(" random numbers between [0, 100] \n");

		printf("sum:      ", sum);
		printNumber(sum, 10);

		printf("\n");
		printf("average:  %.2f \n", average);

		printf("values:   ");
		for(int i = 0; i < 10; i++){
			printf("%i, ", randomValues[i]);
		}
		printf("... \n");

		printf("===========\n");
		printf("#blocks:     %i \n", grid.num_blocks());
		printf("#blocksize:  %i \n", block.num_threads());
	}

}
