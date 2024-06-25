#pragma once

#include "config.h"

static constexpr unsigned int MIN_PER_HOUR = 60;
static constexpr unsigned int MIN_PER_DAY = MIN_PER_HOUR * 24;
static constexpr unsigned int MIN_PER_WEEK = MIN_PER_DAY * 7;
static constexpr unsigned int MIN_PER_YEAR = MIN_PER_WEEK * 52;

struct FormattedTime {
    unsigned int minutes;
    unsigned int hours;
    unsigned int days;
    unsigned int weeks;
    unsigned int years;
};

struct GameTime {
public:
    float realTime_s = 0.0f;
    float dt = 0.0f;

public:
    void incrementRealTime(float dt) {
        this->dt = dt;
        realTime_s += dt;
    }

    float getDt() const { return dt; }

    FormattedTime formattedTime() const {
        uint64_t inGameMinutes = uint64_t(realTime_s / REAL_SECONDS_PER_GAME_MIN);

        FormattedTime gameTime;

        gameTime.years = inGameMinutes / MIN_PER_YEAR;
        inGameMinutes = inGameMinutes % MIN_PER_YEAR;

        gameTime.weeks = inGameMinutes / MIN_PER_WEEK;
        inGameMinutes = inGameMinutes % MIN_PER_WEEK;

        gameTime.days = inGameMinutes / MIN_PER_DAY;
        inGameMinutes = inGameMinutes % MIN_PER_DAY;

        gameTime.hours = inGameMinutes / MIN_PER_HOUR;
        inGameMinutes = inGameMinutes % MIN_PER_HOUR;

        gameTime.minutes = inGameMinutes;

        return gameTime;
    }
};
