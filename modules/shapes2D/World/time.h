#pragma once

#include "config.h"

static constexpr unsigned int MIN_PER_HOUR = 60;
static constexpr unsigned int MIN_PER_DAY = MIN_PER_HOUR * 24;
static constexpr unsigned int MIN_PER_WEEK = MIN_PER_DAY * 7;
static constexpr unsigned int MIN_PER_YEAR = MIN_PER_WEEK * 52;

struct TOD {
    unsigned char hours;
    unsigned char minutes;

    constexpr TOD(unsigned char hours, unsigned char minutes) : hours(hours), minutes(minutes) {}
    constexpr TOD() : hours(0), minutes(0) {}

    void clocktoString(char *timeString) {
        timeString[0] = (hours / 10) + '0';
        timeString[1] = (hours % 10) + '0';
        timeString[2] = ':';
        timeString[3] = (minutes / 10) + '0';
        timeString[4] = (minutes % 10) + '0';
        timeString[5] = 0;
    }

    // day value between 0 and 1 (0 is midnight , 1 midday)
    float toFloat() const {
        if (hours < 12) {
            return float(hours) / 12.0 + float(minutes) / 60.0 / 12.0;
        } else {
            return 1.0f - float(hours - 12) / 12.0 - float(minutes) / 60.0 / 12.0;
        }
    }

    bool operator<(const TOD &other) const { return totalMinutes() < other.totalMinutes(); }

    bool operator<=(const TOD &other) const { return totalMinutes() <= other.totalMinutes(); }

    unsigned int totalMinutes() const { return minutes + hours * 60; }
};

struct TimeInterval {
    TOD start;
    TOD end;

    constexpr TimeInterval(TOD start, TOD end) : start(start), end(end) {}
    constexpr TimeInterval() : start({0, 0}), end({0, 0}) {}

    bool contains(TOD time) const { return start <= time && time <= end; }

    static const TimeInterval factoryHours;
    static const TimeInterval shopHours;
};

/// Used only for display. To perform operations, use GameTime.
struct Date {
    TOD tod;
    unsigned int days;
    unsigned int weeks;
    unsigned int years;
};

struct GameTime {
public:
    float realTime_s = 0.0f;
    float dt = 0.0f;

public:
    GameTime() {}
    GameTime(float realTime_s) : realTime_s(realTime_s) {}
    void incrementRealTime(float dt) {
        this->dt = dt;
        realTime_s += dt;
    }

    float getDt() const { return dt; }

    uint64_t gameMinutes() const { return uint64_t(realTime_s / REAL_SECONDS_PER_GAME_MIN); }

    Date formattedDate() const {
        uint64_t minutes = gameMinutes();

        Date date;

        date.years = minutes / MIN_PER_YEAR;
        minutes = minutes % MIN_PER_YEAR;

        date.weeks = minutes / MIN_PER_WEEK;
        minutes = minutes % MIN_PER_WEEK;

        date.days = minutes / MIN_PER_DAY;

        date.tod = timeOfDay();

        return date;
    }

    TOD timeOfDay() const {
        uint64_t minutes = gameMinutes() % MIN_PER_DAY;

        TOD tod;

        tod.hours = minutes / MIN_PER_HOUR;
        minutes = minutes % MIN_PER_HOUR;

        tod.minutes = minutes;

        return tod;
    }

    unsigned int minutesElapsedSince(const GameTime &previousTime) {
        if (*this < previousTime) {
            printf("previousTime does not happen before the current game time");
        }
        GameTime diff(realTime_s - previousTime.realTime_s);
        return diff.gameMinutes();
    }

    static GameTime fromMinutes(uint32_t minutes) {
        return GameTime(minutes * REAL_SECONDS_PER_GAME_MIN);
    }

    bool operator<(const GameTime &other) { return realTime_s < other.realTime_s; }
    bool operator<=(const GameTime &other) { return realTime_s <= other.realTime_s; }
    GameTime operator+(const GameTime &other) { return GameTime(realTime_s + other.realTime_s); }
};
