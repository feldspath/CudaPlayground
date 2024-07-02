#include "./../common/utils.cuh"
#include "entities.cuh"

uint32_t Entities::newEntity(float2 position, uint32_t house, uint32_t workplace) {
    int32_t id = getCount();
    *count += 1;

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

    return id;
}