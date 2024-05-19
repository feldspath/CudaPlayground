
#pragma once

#define NOMINMAX

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static long long unsuck_start_time =
    std::chrono::high_resolution_clock::now().time_since_epoch().count();

static double Infinity = std::numeric_limits<double>::infinity();

#if defined(__linux__)
constexpr auto fseek_64_all_platforms = fseeko64;
#elif defined(WIN32)
constexpr auto fseek_64_all_platforms = _fseeki64;
#endif

struct MemoryData {
    size_t virtual_total = 0;
    size_t virtual_used = 0;
    size_t virtual_usedByProcess = 0;
    size_t virtual_usedByProcess_max = 0;

    size_t physical_total = 0;
    size_t physical_used = 0;
    size_t physical_usedByProcess = 0;
    size_t physical_usedByProcess_max = 0;
};

struct CpuData {
    double usage = 0;
    size_t numProcessors = 0;
};

MemoryData getMemoryData();

CpuData getCpuData();

void printMemoryReport();

void launchMemoryChecker(int64_t maxMB, double checkInterval);

class punct_facet : public std::numpunct<char> {
protected:
    char do_decimal_point() const { return '.'; };
    char do_thousands_sep() const { return '\''; };
    std::string do_grouping() const { return "\3"; }
};

template <class T> inline std::string formatNumber(T number, int decimals = 0) {
    std::stringstream ss;

    ss.imbue(std::locale(std::cout.getloc(), new punct_facet));
    ss << std::fixed << std::setprecision(decimals);
    ss << number;

    return ss.str();
}

struct Buffer {

    void *data = nullptr;
    uint8_t *data_u8 = nullptr;
    uint16_t *data_u16 = nullptr;
    uint32_t *data_u32 = nullptr;
    uint64_t *data_u64 = nullptr;
    int8_t *data_i8 = nullptr;
    int16_t *data_i16 = nullptr;
    int32_t *data_i32 = nullptr;
    int64_t *data_i64 = nullptr;
    float *data_f32 = nullptr;
    double *data_f64 = nullptr;
    char *data_char = nullptr;

    int64_t id = 0;
    int64_t size = 0;
    int64_t pos = 0;

    Buffer() { this->id = Buffer::createID(); }

    Buffer(int64_t size) {
        data = malloc(size);

        if (data == nullptr) {
            auto memory = getMemoryData();

            std::cout << "ERROR: malloc(" << formatNumber(size) << ") failed.\n";

            auto virtualAvailable = memory.virtual_total - memory.virtual_used;
            auto physicalAvailable = memory.physical_total - memory.physical_used;
            auto GB = 1024.0 * 1024.0 * 1024.0;

            std::cout << "virtual memory(total): "
                      << formatNumber(double(memory.virtual_total) / GB) << '\n';
            std::cout << "virtual memory(used): "
                      << formatNumber(double(memory.virtual_used) / GB, 1) << '\n';
            std::cout << "virtual memory(available): "
                      << formatNumber(double(virtualAvailable) / GB, 1) << '\n';
            std::cout << "virtual memory(used by process): "
                      << formatNumber(double(memory.virtual_usedByProcess) / GB, 1) << '\n';
            std::cout << "virtual memory(highest used by process): "
                      << formatNumber(double(memory.virtual_usedByProcess_max) / GB, 1) << '\n';

            std::cout << "physical memory(total): "
                      << formatNumber(double(memory.physical_total) / GB, 1) << '\n';
            std::cout << "physical memory(available): "
                      << formatNumber(double(physicalAvailable) / GB, 1) << '\n';
            std::cout << "physical memory(used): "
                      << formatNumber(double(memory.physical_used) / GB, 1) << '\n';
            std::cout << "physical memory(used by process): "
                      << formatNumber(double(memory.physical_usedByProcess) / GB, 1) << '\n';
            std::cout << "physical memory(highest used by process): "
                      << formatNumber(double(memory.physical_usedByProcess_max) / GB, 1) << '\n';

            std::cout << "also check if there is enough disk space available" << '\n';

            exit(4312);
        }

        data_u8 = reinterpret_cast<uint8_t *>(data);
        data_u16 = reinterpret_cast<uint16_t *>(data);
        data_u32 = reinterpret_cast<uint32_t *>(data);
        data_u64 = reinterpret_cast<uint64_t *>(data);
        data_i8 = reinterpret_cast<int8_t *>(data);
        data_i16 = reinterpret_cast<int16_t *>(data);
        data_i32 = reinterpret_cast<int32_t *>(data);
        data_i64 = reinterpret_cast<int64_t *>(data);
        data_f32 = reinterpret_cast<float *>(data);
        data_f64 = reinterpret_cast<double *>(data);
        data_char = reinterpret_cast<char *>(data);

        this->size = size;

        this->id = Buffer::createID();
    }

    static int createID() {
        static int64_t counter = 0;

        int id = int(counter);

        counter++;

        return id;
    }

    ~Buffer() { free(data); }

    template <class T> T get(int64_t position) {

        T value;

        memcpy(&value, data_u8 + position, sizeof(T));

        return value;
    }

    template <class T> void set(T value, int64_t position) {
        memcpy(data_u8 + position, &value, sizeof(T));
    }

    inline void write(void *source, int64_t size) {
        memcpy(data_u8 + pos, source, size);

        pos += size;
    }
};

inline double now() {
    auto now = std::chrono::high_resolution_clock::now();
    long long nanosSinceStart = now.time_since_epoch().count() - unsuck_start_time;

    double secondsSinceStart = double(nanosSinceStart) / 1'000'000'000.0;

    return secondsSinceStart;
}

inline void printElapsedTime(std::string label, double startTime) {

    double elapsed = now() - startTime;

    std::string msg = label + ": " + std::to_string(elapsed) + "s\n";
    std::cout << msg << '\n';
}

inline float random(float min, float max) {

    thread_local std::random_device r;
    thread_local std::default_random_engine e(r());

    std::uniform_real_distribution<float> dist(min, max);

    auto value = dist(e);

    return value;
}

inline std::vector<float> random(float min, float max, int n) {

    thread_local std::random_device r;
    thread_local std::default_random_engine e(r());
    std::uniform_real_distribution<float> dist(min, max);

    std::vector<float> values(n);

    for (int i = 0; i < n; i++) {
        auto value = dist(e);
        values[i] = value;
    }

    return values;
}

inline double random(double min, double max) {

    thread_local std::random_device r;
    thread_local std::default_random_engine e(r());

    std::uniform_real_distribution<double> dist(min, max);

    auto value = dist(e);

    return value;
}

inline std::vector<double> random(double min, double max, int n) {

    thread_local std::random_device r;
    thread_local std::default_random_engine e(r());
    std::uniform_real_distribution<double> dist(min, max);

    std::vector<double> values(n);

    for (int i = 0; i < n; i++) {
        auto value = dist(e);
        values[i] = value;
    }

    return values;
}

inline std::vector<int64_t> random(int64_t min, int64_t max, int64_t n) {

    thread_local std::random_device r;
    thread_local std::default_random_engine e(r());
    std::uniform_int_distribution<int64_t> dist(min, max);

    std::vector<int64_t> values(n);

    for (int i = 0; i < n; i++) {
        auto value = dist(e);
        values[i] = value;
    }

    return values;
}

inline std::string stringReplace(std::string str, std::string search, std::string replacement) {

    auto index = str.find(search);

    if (index == str.npos) {
        return str;
    }

    std::string strCopy = str;
    strCopy.replace(index, search.length(), replacement);

    return strCopy;
}

// see https://stackoverflow.com/questions/23943728/case-insensitive-standard-string-comparison-in-c
inline bool icompare_pred(unsigned char a, unsigned char b) {
    return std::tolower(a) == std::tolower(b);
}

// see https://stackoverflow.com/questions/23943728/case-insensitive-standard-string-comparison-in-c
inline bool icompare(std::string const &a, std::string const &b) {
    if (a.length() == b.length()) {
        return std::equal(b.begin(), b.end(), a.begin(), icompare_pred);
    } else {
        return false;
    }
}

inline bool endsWith(const std::string &str, const std::string &suffix) {

    if (str.size() < suffix.size()) {
        return false;
    }

    auto tstr = str.substr(str.size() - suffix.size());

    return tstr.compare(suffix) == 0;
}

inline bool iEndsWith(const std::string &str, const std::string &suffix) {

    if (str.size() < suffix.size()) {
        return false;
    }

    auto tstr = str.substr(str.size() - suffix.size());

    return icompare(tstr, suffix);
}

// taken from:
// https://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring/2602060
inline std::string readTextFile(std::string path) {

    std::ifstream t(path);
    std::string str;

    if (!t.is_open()) {
        std::cerr << "Cannot open file at " << path << '\n';
    }

    t.seekg(0, std::ios::end);
    str.reserve(t.tellg());
    t.seekg(0, std::ios::beg);

    str.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

    return str;
}

// taken from: https://stackoverflow.com/questions/18816126/c-read-the-whole-file-in-buffer
// inline vector<char> readBinaryFile(string path) {
// 	std::ifstream file(path, ios::binary | ios::ate);
// 	std::streamsize size = file.tellg();
// 	file.seekg(0, ios::beg);

// 	std::vector<char> buffer(size);
// 	file.read(buffer.data(), size);

// 	return buffer;
// }

inline std::shared_ptr<Buffer> readBinaryFile(std::string path) {

    auto file = fopen(path.c_str(), "rb");
    auto size = fs::file_size(path);

    // vector<uint8_t> buffer(size);
    auto buffer = std::make_shared<Buffer>(size);

    fread(buffer->data, 1, size, file);
    fclose(file);

    return buffer;
}

// inline vector<uint8_t> readBinaryFile(string path) {
//
//	auto file = fopen(path.c_str(), "rb");
//	auto size = fs::file_size(path);
//
//	vector<uint8_t> buffer(size);
//
//	fread(buffer.data(), 1, size, file);
//	fclose(file);
//
//	return buffer;
// }

//// taken from: https://stackoverflow.com/questions/18816126/c-read-the-whole-file-in-buffer
// inline vector<uint8_t> readBinaryFile(string path, uint64_t start, uint64_t size) {
//	ifstream file(path, ios::binary);
//	//streamsize size = file.tellg();
//
//	auto totalSize = fs::file_size(path);
//
//	if (start >= totalSize) {
//		return vector<uint8_t>();
//	}if (start + size > totalSize) {
//		auto clampedSize = totalSize - start;
//
//		vector<uint8_t> buffer(clampedSize);
//		file.seekg(start, ios::beg);
//		file.read(reinterpret_cast<char*>(buffer.data()), clampedSize);
//
//		return buffer;
//	} else {
//		vector<uint8_t> buffer(size);
//		file.seekg(start, ios::beg);
//		file.read(reinterpret_cast<char*>(buffer.data()), size);
//
//		return buffer;
//	}
// }

inline std::shared_ptr<Buffer> readBinaryFile(std::string path, uint64_t start, uint64_t size) {

    // ifstream file(path, ios::binary);

    // the fopen version seems to be quite a bit faster than ifstream
    auto file = fopen(path.c_str(), "rb");

    auto totalSize = fs::file_size(path);

    if (start >= totalSize) {
        auto buffer = std::make_shared<Buffer>(0);
        return buffer;
    }
    if (start + size > totalSize) {
        auto clampedSize = totalSize - start;

        auto buffer = std::make_shared<Buffer>(clampedSize);
        // file.seekg(start, ios::beg);
        // file.read(reinterpret_cast<char*>(buffer.data()), clampedSize);
        fseek_64_all_platforms(file, start, SEEK_SET);
        fread(buffer->data, 1, clampedSize, file);
        fclose(file);

        return buffer;
    } else {
        auto buffer = std::make_shared<Buffer>(size);
        // file.seekg(start, ios::beg);
        // file.read(reinterpret_cast<char*>(buffer.data()), size);
        fseek_64_all_platforms(file, start, SEEK_SET);
        fread(buffer->data, 1, size, file);
        fclose(file);

        return buffer;
    }
}

inline void readBinaryFile(std::string path, uint64_t start, uint64_t size, void *target) {
    auto file = fopen(path.c_str(), "rb");

    auto totalSize = fs::file_size(path);

    if (start >= totalSize) {
        return;
    }
    if (start + size > totalSize) {
        auto clampedSize = totalSize - start;

        fseek_64_all_platforms(file, start, SEEK_SET);
        fread(target, 1, clampedSize, file);
        fclose(file);
    } else {
        fseek_64_all_platforms(file, start, SEEK_SET);
        fread(target, 1, size, file);
        fclose(file);
    }
}

// writing smaller batches of 1-4MB seems to be faster sometimes?!?
// it's not very significant, though. ~0.94s instead of 0.96s.
template <typename T> inline void writeBinaryFile(std::string path, std::vector<T> &data) {
    std::ios_base::sync_with_stdio(false);
    auto of = std::fstream(path, std::ios::out | std::ios::binary);

    int64_t remaining = data.size() * sizeof(T);
    int64_t offset = 0;

    while (remaining > 0) {
        constexpr int64_t mb4 = int64_t(4 * 1024 * 1024);
        int batchSize = std::min(remaining, mb4);
        of.write(reinterpret_cast<char *>(data.data()) + offset, batchSize);

        offset += batchSize;
        remaining -= batchSize;
    }

    of.close();
}

inline void writeBinaryFile(std::string path, Buffer &data) {
    // std::ios_base::sync_with_stdio(false);
    auto of = std::fstream(path, std::ios::out | std::ios::binary);

    int64_t remaining = data.size;
    int64_t offset = 0;

    while (remaining > 0) {
        constexpr int64_t mb4 = int64_t(4 * 1024 * 1024);
        int batchSize = int(std::min(remaining, mb4));
        of.write(reinterpret_cast<char *>(data.data) + offset, batchSize);

        offset += batchSize;
        remaining -= batchSize;
    }

    of.close();
}

inline void writeBinaryFile(std::string path, uint8_t *data, uint64_t size) {
    // std::ios_base::sync_with_stdio(false);
    auto of = std::fstream(path, std::ios::out | std::ios::binary);

    int64_t remaining = size;
    int64_t offset = 0;

    while (remaining > 0) {
        constexpr int64_t mb4 = int64_t(4 * 1024 * 1024);
        int batchSize = int(std::min(remaining, mb4));
        of.write(reinterpret_cast<char *>(data) + offset, batchSize);

        offset += batchSize;
        remaining -= batchSize;
    }

    of.close();
}

// taken from:
// https://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring/2602060
inline std::string readFile(std::string path) {

    std::ifstream t(path);
    if (!t.is_open()) {
        std::cerr << "Failed to open file at " << path << '\n';
        exit(1);
    }
    std::string str;

    t.seekg(0, std::ios::end);
    str.reserve(t.tellg());
    t.seekg(0, std::ios::beg);

    str.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

    return str;
}

inline void writeFile(std::string path, std::string text) {

    std::ofstream out;
    out.open(path);

    out << text;

    out.close();
}

inline void logDebug(std::string message) {
#if defined(_DEBUG)

    auto id = std::this_thread::get_id();

    std::stringstream ss;
    ss << "[" << id << "]: " << message << "\n";

    std::cout << ss.str();
#endif
}

template <typename T> T read(std::vector<uint8_t> &buffer, int offset) {
    // T value = reinterpret_cast<T*>(buffer.data() + offset)[0];
    T value;

    memcpy(&value, buffer.data() + offset, sizeof(T));

    return value;
}

inline std::string leftPad(std::string in, int length, const char character = ' ') {

    int tmp = int(length - in.size());
    auto reps = std::max(tmp, 0);
    std::string result = std::string(reps, character) + in;

    return result;
}

inline std::string rightPad(std::string in, int64_t length, const char character = ' ') {

    auto reps = std::max(length - int64_t(in.size()), int64_t(0));
    std::string result = in + std::string(reps, character);

    return result;
}

inline std::string repeat(std::string str, int64_t repetitions) {

    std::string result = "";

    for (int i = 0; i < repetitions; i++) {
        result = result + str;
    }

    return result;
}

inline std::vector<std::string> split(std::string str, char delimiter) {

    std::vector<std::string> result;

    int pos = 0;
    while (true) {
        int nextPos = str.find(delimiter, pos);

        if (nextPos == std::string::npos)
            break;

        std::string token = str.substr(pos, nextPos - pos);

        result.push_back(token);

        pos = nextPos + 1;
    }

    {
        std::string token = str.substr(pos, std::string::npos);

        if (token.size() > 0) {
            result.push_back(token);
        }
    }

    return result;
}

void toClipboard(std::string str);

struct EventQueue {

    static EventQueue *instance;
    std::vector<std::function<void()>> queue;
    std::mutex mtx;

    void add(std::function<void()> event) {
        mtx.lock();
        this->queue.push_back(event);
        mtx.unlock();
    }

    void process() {

        mtx.lock();
        std::vector<std::function<void()>> q = queue;
        queue = std::vector<std::function<void()>>();
        mtx.unlock();

        for (auto &event : q) {
            event();
        }
    }
};

inline void schedule(std::function<void()> event) { EventQueue::instance->add(event); }

inline void monitorFile(std::string file, std::function<void()> callback) {

    std::thread([file, callback]() {
        if (!fs::exists(file)) {
            std::cout << "ERROR(monitorFile): file does not exist: " << file << '\n';

            return;
        }

        auto lastWriteTime = fs::last_write_time(fs::path(file));

        using namespace std::chrono_literals;

        while (true) {
            std::this_thread::sleep_for(20ms);

            auto currentWriteTime = fs::last_write_time(fs::path(file));

            if (currentWriteTime > lastWriteTime) {

                // callback();
                schedule(callback);

                lastWriteTime = currentWriteTime;
            }
        }
    }).detach();
}

#define GENERATE_ERROR_MESSAGE std::cout << "ERROR(" << __FILE__ << ":" << __LINE__ << "): "
#define GENERATE_WARN_MESSAGE  std::cout << "WARNING: "
