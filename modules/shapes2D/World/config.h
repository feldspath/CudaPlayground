#pragma once

static constexpr int MAPX = 64;
static constexpr int MAPY = 64;

static int FACTORY_CAPACITY = 4;
static int SHOP_WORK_CAPACITY = 2;
static int SHOP_CUSTOMERS_PER_WORKER = 4;
#define HOUSE_UPGRADE_WOOD_COUNT 20
#define str(x)                   #x
#define xstr(x)                  str(x)

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

static float REAL_SECONDS_PER_GAME_MIN = 60.0 / (60.0 * 24.0);
