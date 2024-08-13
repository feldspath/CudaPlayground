#pragma once

static constexpr int MAPX = 64;
static constexpr int MAPY = 64;

static int FACTORY_CAPACITY = 40;
static int SHOP_WORK_CAPACITY = 40;
static int SHOP_CUSTOMERS_PER_WORKER = 4;
static int HOUSE_BASE_UPGRADE_WOOD_COUNT = 5;
static int WOOD_SELL_PRICE = 5;
static int SHOP_WORKER_INVENTORY_SIZE = 40;

static float ENTITY_RADIUS = 0.2f;
static float ENTITY_SPEED = 10.0f;

static float CELL_RADIUS = 0.5f;
static float CELL_PADDING = 0.0f;

static uint32_t SHOP_TIME_MIN = 60;

static float KERNEL_RADIUS = 1.0f;
static float REPULSIVE_STRENGTH = 30.0f;
static float STIR_STRENGTH = 20.0f;
static float DAMPING_STRENGTH = 10.0f;

static unsigned int ROAD_COST = 5;
static unsigned int FACTORY_COST = 100;
static unsigned int HOUSE_COST = 50;

static float REAL_SECONDS_PER_GAME_MIN = 300.0 / (60.0 * 24.0);

// Let's assume we can have 10 times as much entities as we have cells (40k for now)
static constexpr int MAX_ENTITY_COUNT = 10 * MAPX * MAPY;

static constexpr int MAX_FLOWFIELDS_PER_FRAME = 200;
static constexpr int MAX_PATHS_PER_FRAME = 4000;
