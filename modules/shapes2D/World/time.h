#pragma once

#include "config.h"

static constexpr unsigned int MIN_PER_HOUR = 60;
static constexpr unsigned int MIN_PER_DAY = MIN_PER_HOUR * 24;
static constexpr unsigned int MIN_PER_WEEK = MIN_PER_DAY * 7;
static constexpr unsigned int MIN_PER_YEAR = MIN_PER_WEEK * 52;

struct FormattedTOD {
    unsigned char hours;
    unsigned char minutes;

    constexpr FormattedTOD(unsigned char hours, unsigned char minutes)
        : hours(hours), minutes(minutes) {}
    constexpr FormattedTOD() : hours(0), minutes(0) {}

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

    bool operator<(const FormattedTOD &other) const {
        return hours < other.hours || (hours == other.hours && minutes < other.minutes);
    }

    bool operator<=(const FormattedTOD &other) const {
        return hours < other.hours || (hours == other.hours && minutes <= other.minutes);
    }

    FormattedTOD operator-(const FormattedTOD &other) const {
        if (*this < other) {
            printf("invalid date substration");
        }
        return FormattedTOD(hours - other.hours, minutes - other.minutes);
    }

    unsigned int totalMinutes() const { return minutes + hours * 60; }
};

struct TimeInterval {
    FormattedTOD start;
    FormattedTOD end;

    constexpr TimeInterval(FormattedTOD start, FormattedTOD end) : start(start), end(end) {}
    constexpr TimeInterval() : start({0, 0}), end({0, 0}) {}

    bool contains(FormattedTOD time) const { return start <= time && time <= end; }

    static const TimeInterval factoryHours;
    static const TimeInterval shopHours;
};

struct FormattedDate {
    FormattedTOD tod;
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

    FormattedDate formattedDate() const {
        uint64_t inGameMinutes = uint64_t(realTime_s / REAL_SECONDS_PER_GAME_MIN);

        FormattedDate date;

        date.years = inGameMinutes / MIN_PER_YEAR;
        inGameMinutes = inGameMinutes % MIN_PER_YEAR;

        date.weeks = inGameMinutes / MIN_PER_WEEK;
        inGameMinutes = inGameMinutes % MIN_PER_WEEK;

        date.days = inGameMinutes / MIN_PER_DAY;
        inGameMinutes = inGameMinutes % MIN_PER_DAY;

        date.tod = formattedTime();

        return date;
    }

    FormattedTOD formattedTime() const {
        uint64_t inGameMinutes = uint64_t(realTime_s / REAL_SECONDS_PER_GAME_MIN);
        inGameMinutes = inGameMinutes % MIN_PER_DAY;

        FormattedTOD tod;

        tod.hours = inGameMinutes / MIN_PER_HOUR;
        inGameMinutes = inGameMinutes % MIN_PER_HOUR;

        tod.minutes = inGameMinutes;

        return tod;
    }
};
