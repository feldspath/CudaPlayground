
#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "../common/utils.cuh"

namespace cg = cooperative_groups;

constexpr int numElements = 100;
// constexpr int numElements = 5'000'000;
constexpr int maxValue = 1000;

constexpr bool TRACE = true;

void radix_sort(
	uint64_t* numbers, uint32_t count,
	uint32_t* globalCounters, uint32_t* globalCounters_blocks, 
	uint32_t* globalPrefixSum, uint32_t* globalPrefixSums_blocks,
	uint64_t* intermediateStorage
){

	// sort in multiple chunks of bits, e.g. 10 bits -> 2^10 = 1024 bins -> fits in workgroup memory
	// start with least-significant bits. 
	// must be stable if sorted from least to most significant bits?

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	uint64_t t_start = nanotime();
	

	constexpr int64_t bitsize = 8;
	constexpr int64_t bitmask = (1 << bitsize) - 1;
	constexpr int64_t numBins = 1 << bitsize;

	__shared__ uint32_t sh_bin_counters[numBins];
	__shared__ uint32_t sh_bin_prefixes[numBins];

	int currentPass = 0;

	// reset local bin counters
	for(
		int i = block.thread_rank(); 
		i < numBins; 
		i += block.num_threads()
	){
		sh_bin_counters[i] = 0;
	}

	grid.sync();

	// reset global bin counters
	processRange(numBins, [&](int index){
		globalCounters[index] = 0;
	});
	

	grid.sync();

	// count in shared memory
	processRange(count, [&](int index){
		int64_t number = numbers[index];

		uint64_t binIndex = (number >> (currentPass * bitsize)) & bitmask;

		atomicAdd(&sh_bin_counters[binIndex], 1);
	});
	grid.sync();

	// merge counts to global counter
	for(
		int i = block.thread_rank(); 
		i < numBins; 
		i += block.num_threads()
	){
		atomicAdd(&globalCounters[i], sh_bin_counters[i]);
	}

	// shared to global block counters
	for(
		int i = block.thread_rank(); 
		i < numBins; 
		i += block.num_threads()
	){
		globalCounters_blocks[grid.block_rank() * numBins + i] = sh_bin_counters[i];
	}


	grid.sync();

	// compute global prefix sum
	if(grid.thread_rank() == 0){

		int sum = 0;
		for(int i = 0; i < numBins; i++){
			globalPrefixSum[i] = sum;
			sum = sum + globalCounters[i];
		}
	}

	grid.sync();

	// compute blockwise prefix sum
	for(
		int binIndex = block.thread_rank(); 
		binIndex < numBins; 
		binIndex += block.num_threads()
	){
		int sum = 0;
		for(int blockIndex = 0; blockIndex < grid.num_blocks(); blockIndex++){
			
			globalPrefixSums_blocks[blockIndex * numBins + binIndex] = globalPrefixSum[binIndex] + sum;
			int blockcounter = globalCounters_blocks[blockIndex * numBins + binIndex];

			sum = sum + blockcounter;
		}

		globalCounters[binIndex] = sum;
	}


	grid.sync();

	// // DEBUG: print counters
	if(TRACE)
	if(grid.thread_rank() == 0){

		printf("binIndex   offset    count\n");
		printf("======================================================================================================================\n");
		for(int binIndex = 0; binIndex <= min(20llu, numBins); binIndex++){

			if(binIndex == 20){
				binIndex = numBins - 1;
				printf("...\n");
			}

			int count = globalCounters[binIndex];
			int offset = globalPrefixSum[binIndex];
			
			printf("[%6u] %8u  %7u   ", binIndex, offset, count);

			for(int blockIndex = 0; blockIndex < 5; blockIndex++){
				// int value = globalPrefixSums_blocks[grid.block_rank() * numBins + binIndex];
				int counter = globalCounters_blocks[blockIndex * numBins + binIndex];
				int prefix = globalPrefixSums_blocks[blockIndex * numBins + binIndex];

				printf("| %6u %6u ", prefix, counter);
			}
			printf("\n");
		}
		// if(numBins > 20){
		// 	printf(".\n");
		// 	printf("[%6u]\n", numBins);
		// }
		printf("======================================================================================================================\n");
	}

	grid.sync();

	// if(grid.thread_rank() == 0){
	// 	uint64_t nanos = nanotime() - t_start;
	// 	double millies = double(nanos) / 1'000'000.0;
	// 	printf("===\n");
	// 	printf("radix_sort duration: %.1f ms \n", millies);
	// }

	// store values in target bin
	processRange(count, [&](int index){
		int64_t number = numbers[index];

		uint64_t binIndex = (number >> (currentPass * bitsize)) & bitmask;

		// if(number == 921){
		// 	printf("%llu, ", binIndex);
		// }

		// int targetIndex = atomicAdd(&globalPrefixSum[binIndex], 1);
		// int targetIndex = globalPrefixSum[binIndex];

		uint32_t srcIndex = grid.block_rank() * numBins + binIndex;
		int targetIndex = atomicAdd(&globalPrefixSums_blocks[srcIndex], 1);

		// int targetIndex = index;
		intermediateStorage[targetIndex] = number;
	});

	grid.sync();

	// if(grid.thread_rank() == 0){
	// 	uint64_t nanos = nanotime() - t_start;
	// 	double millies = double(nanos) / 1'000'000.0;

	// 	printf("radix_sort duration: %.1f ms \n", millies);
	// }



	// print some sorted values
	if(TRACE)
	if(grid.thread_rank() == 0){

		int stride = numElements / 10;
		int valuesPerStride = 10;

		for(int i = 0; i < numElements; i += stride){

			printf("[%8u] ", i);
			
			for(int j = 0; j < valuesPerStride; j++){
				int64_t value = intermediateStorage[i + j];

				printf("%llu, ", value);
			}
			printf("\n");
		}
	}


	// if(grid.thread_rank() == 0){
	// 	uint64_t nanos = nanotime() - t_start;
	// 	double millies = double(nanos) / 1'000'000.0;

	// 	printf("radix_sort duration: %.1f ms \n", millies);
	// }


	

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

	// store random values in array
	processRange(0, numElements, [&](int index){
		uint64_t randomValue = curand(&thread_random_state);
		uint64_t random_0_100 = randomValue % (maxValue + 1);
		randomValues[index] = random_0_100;
	});

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



	uint64_t* intermediateStorage = allocator.alloc<uint64_t*>(8 * numElements); 

	uint64_t t_start = nanotime();
	grid.sync();
	for(int i = 0; i < 1; i++){
		
		radix_sort(randomValues, numElements, 
			globalCounters, globalCounters_blocks, 
			globalPrefixSums, globalPrefixSums_blocks, 
			intermediateStorage);
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
