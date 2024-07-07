#pragma once

struct Particles{

	bool initialized;
	int capacity;

	float spawnsPerSecond;
	float accumulated_time;

	float max_age;

	int mutex;
	int particleCounter;
	int* availableParticles;

	float2* position;
	float2* size;
	float2* velocity;
	float* age;

	void init();

	void acquireLock();

	void releaseLock();

	int spawn();

	void despawn(int index);

};