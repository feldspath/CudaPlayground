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

    void clocktoString(char *timeString) {
        timeString[0] = (hours / 10) + '0';
        timeString[1] = (hours % 10) + '0';
        timeString[2] = ':';
        timeString[3] = (minutes / 10) + '0';
        timeString[4] = (minutes % 10) + '0';
        timeString[5] = 0;
    }

    // day value between 0 and 1 (0 is midnight , 1 midday)
    float timeOfDay() const {
        if (hours < 12) {
            return float(hours) / 12.0 + float(minutes) / 60.0 / 12.0;
        } else {
            return 1.0f - float(hours - 12) / 12.0 - float(minutes) / 60.0 / 12.0;
        }
    }
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
