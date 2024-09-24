#pragma once

static constexpr int CHUNK_X = 64;
static constexpr int CHUNK_Y = 64;
static constexpr int N_CHUNK_X = 4;
static constexpr int N_CHUNK_Y = 4;
static constexpr int CHUNK_SIZE = CHUNK_X * CHUNK_Y;

static int FACTORY_CAPACITY = 40;
static int SHOP_WORK_CAPACITY = 40;
static int SHOP_CUSTOMERS_PER_WORKER = 4;
static int HOUSE_BASE_UPGRADE_WOOD_COUNT = 5;
static int WOOD_SELL_PRICE = 5;
static int SHOP_WORKER_INVENTORY_SIZE = 40;

static float ENTITY_RADIUS = 0.2f;
static float ENTITY_SPEED = 10.0f;

static float CELL_RADIUS = 0.5f;

static uint32_t SHOP_TIME_MIN = 60;

static float KERNEL_RADIUS = 1.0f;
static float REPULSIVE_STRENGTH = 30.0f;
static float STIR_STRENGTH = 20.0f;
static float DAMPING_STRENGTH = 10.0f;

static unsigned int ROAD_COST = 5;
static unsigned int FACTORY_COST = 100;
static unsigned int HOUSE_COST = 50;

static float REAL_SECONDS_PER_GAME_MIN = 300.0 / (60.0 * 24.0);

// Target entity count
static constexpr int MAX_ENTITY_COUNT = 1'000'000;

static constexpr int MAX_PATHS_PER_FRAME = 10000;
