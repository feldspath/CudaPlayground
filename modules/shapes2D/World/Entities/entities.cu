#include "./../common/utils.cuh"
#include "entities.cuh"

uint32_t Entities::newEntity(float2 position, MapId house, MapId workplace) {
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
    entity.house = house;
    entity.workplace = workplace;
    entity.state = Rest;
    entity.path.reset();
    entity.destination = MapId::invalidId();
    entity.interaction = -1;
    entity.active = true;
    entity.disabled = false;
    entity.inventory = 0;

    return id;
}