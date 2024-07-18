#include "./../common/utils.cuh"
#include "entities.cuh"

uint32_t Entities::newEntity(float2 position, uint32_t house, uint32_t workplace) {
    int32_t id;
    if (*holesCount == 0) {
        id = getCount();
        *count += 1;
    } else {
        id = *((uint32_t *)(&buffer[MAX_ENTITY_COUNT]) - *holesCount);
        (*holesCount)--;
    }

    Entity &entity = get(id);
    entity.position = position;
    entity.velocity = {0.0f, 0.0f};
    entity.houseId = house;
    entity.workplaceId = workplace;
    entity.state = Rest;
    entity.path.reset();
    entity.money = 0;
    entity.destination = -1;
    entity.interaction = -1;
    entity.happiness = 255;
    entity.active = true;
    entity.inventory = 0;

    return id;
}