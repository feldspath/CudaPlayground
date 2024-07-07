
#include "./Rendering/Particles.h"

#include <cooperative_groups.h>
#include "common/utils.cuh"

constexpr int MTX_FREE = 0;
constexpr int MTX_LOCKED = 1;

void Particles::init(){
	if(this->initialized) return;

	auto grid = cg::this_grid();

	processRange(capacity, [&](int index){
		this->availableParticles[index] = index;
	});

	grid.sync();

	this->initialized = true;
	this->accumulated_time = 0.0f;
}

void Particles::acquireLock(){

	while(!atomicCAS(&this->mutex, MTX_FREE, MTX_LOCKED)){
		__nanosleep(1);		
	}

}

void Particles::releaseLock(){
	atomicExch(&this->mutex, MTX_FREE);
}

int Particles::spawn(){

	this->acquireLock();

	if(particleCounter + 1 < capacity){
		int index = atomicAdd(&particleCounter, 1);

		// printf("availableParticles[%i] = %i \n", index, availableParticles[index]);

		int particleIndex = availableParticles[index];

		this->releaseLock();

		return particleIndex;
	}else{
		this->releaseLock();

		return -1;
	}
	
}

void Particles::despawn(int index){

	this->acquireLock();

	atomicAdd(&particleCounter, -1);

	availableParticles[particleCounter] = index;

	this->age[index] = -1.0f;

	this->releaseLock();
}
