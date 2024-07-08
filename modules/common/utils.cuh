
#pragma once

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define FALSE 0
#define TRUE  1

typedef unsigned int uint32_t;
typedef int int32_t;
// typedef char int8_t;
typedef unsigned char uint8_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;

#define Infinity 0x7f800000

// calls function <f> <size> times
// calls are distributed over all available threads
template <typename Function> void processRange(int first, int size, Function &&f) {

    uint32_t totalThreadCount = blockDim.x * gridDim.x;

    int itemsPerThread = size / totalThreadCount + 1;

    for (int i = 0; i < itemsPerThread; i++) {
        int block_offset = itemsPerThread * blockIdx.x * blockDim.x;
        int thread_offset = itemsPerThread * threadIdx.x;
        int index = first + block_offset + thread_offset + i;

        if (index >= first + size) {
            break;
        }

        f(index);
    }
}

template <typename Function> void processRange(int size, Function &&f) {

    uint32_t totalThreadCount = blockDim.x * gridDim.x;

    int itemsPerThread = size / totalThreadCount + 1;

    for (int i = 0; i < itemsPerThread; i++) {
        int block_offset = itemsPerThread * blockIdx.x * blockDim.x;
        int thread_offset = itemsPerThread * threadIdx.x;
        int index = block_offset + thread_offset + i;

        if (index >= size) {
            break;
        }

        f(index);
    }
}

// calls function <f> <size> times
// calls are distributed over all threads in a block.
template <typename Function> inline void processRangeBlock(int size, Function &&f) {

    uint32_t totalThreadCount = blockDim.x;

    int itemsPerThread = size / totalThreadCount + 1;

    for (int i = 0; i < itemsPerThread; i++) {
        int thread_offset = threadIdx.x;
        int index = totalThreadCount * i + thread_offset;

        if (index >= size) {
            break;
        }

        f(index);
    }
}

// Loops through [0, size), but blockwise instead of threadwise.
// That is, all threads of block 0 are called with index 0, block 1 with index 1, etc.
// Intented for when <size> is larger than the number of blocks,
// e.g., size 10'000 but #blocks only 100, then the blocks will keep looping until all indices are
// processed.
inline int for_blockwise_counter;
template <typename Function> inline void for_blockwise(int size, Function &&f) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    __shared__ int sh_index;
    sh_index = 0;
    for_blockwise_counter = 0;

    grid.sync();

    while (true) {

        if (block.thread_rank() == 0) {
            uint32_t index = atomicAdd(&for_blockwise_counter, 1);
            sh_index = index;
        }

        block.sync();

        if (sh_index >= size)
            break;

        f(sh_index);
    }
}

void printNumber(int64_t number, int leftPad = 0);

struct Allocator {

    uint8_t *buffer = nullptr;
    int64_t offset = 0;

    template <class T> Allocator(T buffer) {
        this->buffer = reinterpret_cast<uint8_t *>(buffer);
        this->offset = 0;
    }

    Allocator(unsigned int *buffer, int64_t offset) {
        this->buffer = reinterpret_cast<uint8_t *>(buffer);
        this->offset = offset;
    }

    template <class T> T alloc(int64_t size) {

        auto ptr = reinterpret_cast<T>(buffer + offset);

        int64_t newOffset = offset + size;

        // make allocated buffer location 16-byte aligned to avoid
        // potential problems with bad alignments
        int64_t remainder = (newOffset % 16ll);

        if (remainder != 0ll) {
            newOffset = (newOffset - remainder) + 16ll;
        }

        this->offset = newOffset;

        return ptr;
    }

    template <class T> T alloc(int64_t size, const char *label) {

        // if(isFirstThread()){
        // 	printf("offset: ");
        // 	printNumber(offset, 13);
        // 	printf(", allocating: ");
        // 	printNumber(size, 13);
        // 	printf(", label: %s \n", label);
        // }

        auto ptr = reinterpret_cast<T>(buffer + offset);

        int64_t newOffset = offset + size;

        // make allocated buffer location 16-byte aligned to avoid
        // potential problems with bad alignments
        int64_t remainder = (newOffset % 16ll);

        if (remainder != 0ll) {
            newOffset = (newOffset - remainder) + 16ll;
        }

        this->offset = newOffset;

        return ptr;
    }
};

inline uint64_t nanotime() {

    uint64_t nanotime;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanotime));

    return nanotime;
}

inline int strlen(const char *text) {
    int MAX_STRLEN = 100;
    int numchars = 0;

    for (int i = 0; i < MAX_STRLEN; i++) {
        if (text[i] == 0)
            break;

        numchars++;
    }

    return numchars;
}

inline uint32_t rgb8color(float3 color) {
    uint32_t r = color.x * 255.0f;
    uint32_t g = color.y * 255.0f;
    uint32_t b = color.z * 255.0f;
    uint32_t rgb8color = r | (g << 8) | (b << 16);
    return rgb8color;
}

inline uint32_t rgba8color(float4 color) {
    uint32_t r = color.x * 255.0f;
    uint32_t g = color.y * 255.0f;
    uint32_t b = color.z * 255.0f;
    uint32_t a = color.w * 255.0f;
    uint32_t rgb8color = r | (g << 8) | (b << 16) | (a << 24);
    return rgb8color;
}

inline float3 float3color(uint32_t color) {
    uint32_t r = color & 0xFF;
    uint32_t g = (color >> 8) & 0xFF;
    uint32_t b = (color >> 16) & 0xFF;
    return float3{float(r) / 255.0f, float(g) / 255.0f, float(b) / 255.0f};
}

inline float4 float4color(uint32_t color) {
    uint32_t r = color & 0xFF;
    uint32_t g = (color >> 8) & 0xFF;
    uint32_t b = (color >> 16) & 0xFF;
    uint32_t a = (color >> 24) & 0xFF;
    return float4{float(r) / 255.0f, float(g) / 255.0f, float(b) / 255.0f, float(a) / 255.0f};
}

#define MAX_STRING_LENGTH 16

static void itos(unsigned int value, char *string) {
    char chars[MAX_STRING_LENGTH];
    int i;
    for (i = 0; i < 10; ++i) {
        if (i > 0 && value == 0) {
            break;
        }
        chars[i] = value % 10 + '0';
        value = value / 10;
    }

    for (int j = 0; j < i; j++) {
        string[i - 1 - j] = chars[j];
    }
    string[i] = 0;
}

static void strcpy(char *dest, const char *src) {
    int i = 0;
    do {
        dest[i] = src[i];
    } while (src[i++] != 0);
}

static void strcat(char *dest, const char *src) {
    int i = 0;
    while (dest[i] != 0)
        i++;
    strcpy(dest + i, src);
}