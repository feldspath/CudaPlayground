#pragma once

#include "cuda.h"
#include "nvrtc.h"
#include "unsuck.hpp"
#include <cmath>
#include <nvJitLink.h>
#include <print>
#include <string>
#include <unordered_map>

using std::string;

#define NVJITLINK_SAFE_CALL(h, x)                                                                  \
    do {                                                                                           \
        nvJitLinkResult result = x;                                                                \
        if (result != NVJITLINK_SUCCESS) {                                                         \
            std::cerr << "\nerror: " #x " failed with error " << result << '\n';                   \
            size_t lsize;                                                                          \
            result = nvJitLinkGetErrorLogSize(h, &lsize);                                          \
            if (result == NVJITLINK_SUCCESS && lsize > 0) {                                        \
                char *log = (char *)malloc(lsize);                                                 \
                result = nvJitLinkGetErrorLog(h, log);                                             \
                if (result == NVJITLINK_SUCCESS) {                                                 \
                    std::cerr << "error: " << log << '\n';                                         \
                    free(log);                                                                     \
                }                                                                                  \
            }                                                                                      \
            exit(1);                                                                               \
        } else {                                                                                   \
            size_t lsize;                                                                          \
            result = nvJitLinkGetInfoLogSize(h, &lsize);                                           \
            if (result == NVJITLINK_SUCCESS && lsize > 0) {                                        \
                char *log = (char *)malloc(lsize);                                                 \
                result = nvJitLinkGetInfoLog(h, log);                                              \
                if (result == NVJITLINK_SUCCESS) {                                                 \
                    std::cerr << "info: " << log << '\n';                                          \
                    free(log);                                                                     \
                }                                                                                  \
            }                                                                                      \
            break;                                                                                 \
        }                                                                                          \
    } while (0)

struct OptionalLaunchSettings {
    uint32_t gridsize = 0;
    uint32_t blocksize = 0;
    std::vector<void *> args;
    bool measureDuration = false;
};

// void printInfoLog(nvJitLinkHandle handle){
// 	size_t infoLogSize;
// 	nvJitLinkGetInfoLogSize(handle, &infoLogSize);

// 	if (infoLogSize > 0) {
// 		char *log = (char*)malloc(infoLogSize);
// 		nvJitLinkGetInfoLog(handle, log);

// 		// stringstream ss;
// 		cout << "INFO: " << log << endl;

// 		free(log);

// 		// return ss.str();
// 	}

// 	// return "";
// }

struct CudaModule {

    void cu_checked(CUresult result) {
        if (result != CUDA_SUCCESS) {
            std::cout << "cuda error code: " << result << std::endl;
        }
    };

    string path = "";
    string name = "";
    bool compiled = false;
    bool success = false;

    size_t ptxSize = 0;
    char *ptx = nullptr;

    size_t ltoirSize = 0;
    char *ltoir = nullptr;

    // size_t nvvmSize;
    // char *nvvm = nullptr;

    std::vector<std::string> customIncludeDirs;

    CudaModule(string path, string name,
               std::vector<std::string> includeDirs = std::vector<std::string>()) {
        this->path = path;
        this->name = name;
        this->customIncludeDirs = includeDirs;
    }

    void compile() {
        auto tStart = now();

        std::cout
            << "================================================================================"
            << std::endl;
        std::cout << "=== COMPILING: " << fs::path(path).filename().string() << std::endl;
        std::cout
            << "================================================================================"
            << std::endl;

        success = false;

        string dir = fs::path(path).parent_path().string();
        // string optInclude = "-I " + dir;

        string cuda_path = std::getenv("CUDA_PATH");
        // string cuda_include = "-I " + cuda_path + "/include";

        string optInclude = std::format("-I {}", dir).c_str();
        string cuda_include = std::format("-I {}/include", cuda_path);
        string cudastd_include = std::format("-I {}/include/cuda/std", cuda_path);
        string cudastd_detail_include =
            std::format("-I {}/include/cuda/std/detail/libcxx/include", cuda_path);
        string wtf = "-I C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/ucrt";
        string wtf2 = "-I C:/Program Files/Microsoft Visual "
                      "Studio/2022/Community/VC/Tools/MSVC/14.36.32532/include";

        //"C:\Program Files\NVIDIA GPU Computing
        // Toolkit\CUDA\v11.8\include\cuda\std\detail\libcxx\include\iterator"

        // string i_cub            = format("-I {}",
        // "D:/dev/workspaces/CudaPlayground/gaussian_private/libs/cccl-main/cub"); string i_libcuda
        // = format("-I {}",
        // "D:/dev/workspaces/CudaPlayground/gaussian_private/libs/cccl-main/libcudacxx/include");
        // string i_cudastd_detail = format("-I {}",
        // "D:/dev/workspaces/CudaPlayground/gaussian_private/libs/cccl-main/libcudacxx/include/cuda/std/detail/libcxx/include/");
        // string i_libcudastd     = format("-I {}",
        // "D:/dev/workspaces/CudaPlayground/gaussian_private/libs/cccl-main/libcudacxx/include/cuda/std");

        std::vector<std::string> i_customs;
        for (auto &dir : customIncludeDirs) {
            i_customs.push_back(std::format(" -I {}", dir));
        }

        nvrtcProgram prog;
        string source = readFile(path);
        nvrtcCreateProgram(&prog, source.c_str(), name.c_str(), 0, NULL, NULL);
        std::vector<const char *> opts = {
            "--gpu-architecture=compute_75",
            // "--gpu-architecture=compute_86",
            "--use_fast_math", "--extra-device-vectorization", "-lineinfo",
            // i_cub.c_str(),
            // i_libcuda.c_str(),
            // i_libcudastd.c_str(),
            cudastd_include.c_str(), cuda_include.c_str(), optInclude.c_str(), "-I ./",
            // wtf.c_str(),
            // wtf2.c_str(),
            "--relocatable-device-code=true",
            "-default-device",  // assume __device__ if not specified
            "--dlink-time-opt", // link time optimization "-dlto",
            // "--dopt=on",
            "--std=c++20", "--disable-warnings",
            "--split-compile=0", // compiler optimizations in parallel. 0 -> max available threads
            "--time=cuda_compile_time.txt", // show compiler timings
        };

        for (auto &i : i_customs) {
            opts.push_back(i.c_str());
        }

        for (auto opt : opts) {
            std::cout << opt << std::endl;
        }
        std::cout << "====" << std::endl;

        nvrtcResult res = nvrtcCompileProgram(prog, opts.size(), opts.data());

        if (res != NVRTC_SUCCESS) {
            size_t logSize;
            nvrtcGetProgramLogSize(prog, &logSize);
            char *log = new char[logSize];
            nvrtcGetProgramLog(prog, log);
            std::cerr << "Program Log: " << log << std::endl;

            delete[] log;

            if (res != NVRTC_SUCCESS && ltoir != nullptr) {
                return;
            } else if (res != NVRTC_SUCCESS && ltoir == nullptr) {
                exit(123);
            }
        }

        // if(nvvmSize > 0){
        //	delete[] nvvm;
        //	nvvmSize = 0;
        // }

        nvrtcGetLTOIRSize(prog, &ltoirSize);
        ltoir = new char[ltoirSize];
        nvrtcGetLTOIR(prog, ltoir);

        std::cout << std::format("compiled ltoir. size: {} byte \n", ltoirSize);

        // nvrtcGetNVVMSize(prog, &nvvmSize);
        // nvvm = new char[nvvmSize];
        // nvrtcGetNVVM(prog, nvvm);
        //// Destroy the program.
        nvrtcDestroyProgram(&prog);

        compiled = true;
        success = true;

        printElapsedTime("compile " + name, tStart);
    }
};

struct CudaModularProgram {

    struct CudaModularProgramArgs {
        std::vector<string> modules;
        std::vector<std::string> customIncludeDirs;
    };

    void cu_checked(CUresult result) {
        if (result != CUDA_SUCCESS) {
            std::cout << "cuda error code: " << result << std::endl;
        }
    };

    std::vector<CudaModule *> modules;

    CUmodule mod;
    // CUfunction kernel = nullptr;
    void *cubin;
    size_t cubinSize;

    std::vector<std::function<void(void)>> compileCallbacks;

    std::vector<string> kernelNames;
    std::unordered_map<string, CUfunction> kernels;

    std::unordered_map<string, CUevent> events_launch_start;
    std::unordered_map<string, CUevent> events_launch_end;
    std::unordered_map<string, float> last_launch_duration;

    CudaModularProgram(CudaModularProgramArgs args) {
        // CudaModularProgram(vector<string> modulePaths, vector<string> kernelNames = {}){

        std::vector<string> modulePaths = args.modules;
        // vector<string> kernelNames = args.kernels;

        // this->kernelNames = kernelNames;

        for (auto modulePath : modulePaths) {

            string moduleName = fs::path(modulePath).filename().string();
            auto module = new CudaModule(modulePath, moduleName, args.customIncludeDirs);

            module->compile();

            monitorFile(modulePath, [&, module]() {
                module->compile();
                link();
            });

            modules.push_back(module);
        }

        link();
    }

    void link() {

        std::cout
            << "================================================================================"
            << std::endl;
        std::cout << "=== LINKING" << std::endl;
        std::cout
            << "================================================================================"
            << std::endl;

        auto tStart = now();

        for (auto module : modules) {
            if (!module->success) {
                return;
            }
        }

        float walltime;
        constexpr uint32_t v_optimization_level = 1;
        constexpr uint32_t logSize = 8192;
        char info_log[logSize];
        char error_log[logSize];

        // vector<CUjit_option> options = {
        // 	CU_JIT_LTO,
        // 	CU_JIT_WALL_TIME,
        // 	CU_JIT_OPTIMIZATION_LEVEL,
        // 	CU_JIT_INFO_LOG_BUFFER,
        // 	CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        // 	CU_JIT_ERROR_LOG_BUFFER,
        // 	CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        // 	CU_JIT_LOG_VERBOSE,
        // 	// CU_JIT_FAST_COMPILE // CUDA internal only (?)
        // };

        // vector<void*> optionVals = {
        // 	(void*) 1,
        // 	(void*) &walltime,
        // 	(void*) 4,
        // 	(void*) info_log,
        // 	(void*) logSize,
        // 	(void*) error_log,
        // 	(void*) logSize,
        // 	(void*) 1,
        // 	// (void*) 1
        // };

        CUlinkState linkState;

        CUdevice cuDevice;
        cuDeviceGet(&cuDevice, 0);

        int major = 0;
        int minor = 0;
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);

        int arch = major * 10 + minor;
        // char smbuf[16];
        // memset(smbuf, 0, 16);
        // sprintf(smbuf, "-arch=sm_%d\n", arch);

        string strArch = std::format("-arch=sm_{}", arch);

        const char *lopts[] = {
            "-dlto", // link time optimization
            strArch.c_str(),
            "-time",
            "-verbose",
            "-O3", // optimization level
            "-optimize-unused-variables",
            "-split-compile=0",
        };

        nvJitLinkHandle handle;
        nvJitLinkCreate(&handle, 2, lopts);

        for (auto module : modules) {
            NVJITLINK_SAFE_CALL(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR,
                                                         (void *)module->ltoir, module->ltoirSize,
                                                         module->name.c_str()));
        }

        NVJITLINK_SAFE_CALL(handle, nvJitLinkComplete(handle));
        NVJITLINK_SAFE_CALL(handle, nvJitLinkGetLinkedCubinSize(handle, &cubinSize));

        cubin = malloc(cubinSize);
        NVJITLINK_SAFE_CALL(handle, nvJitLinkGetLinkedCubin(handle, cubin));
        NVJITLINK_SAFE_CALL(handle, nvJitLinkDestroy(&handle));

        static int cubinID = 0;
        // writeBinaryFile(format("./program_{}.cubin", cubinID), (uint8_t*)cubin, cubinSize);
        cubinID++;

        cu_checked(cuModuleLoadData(&mod, cubin));

        { // Retrieve Kernels
            uint32_t count = 0;
            cuModuleGetFunctionCount(&count, mod);

            std::vector<CUfunction> functions(count);
            cuModuleEnumerateFunctions(functions.data(), count, mod);

            kernelNames.clear();

            for (CUfunction function : functions) {
                const char *name;
                CUkernel kernel = (CUkernel)function;
                cuFuncGetName(&name, function);

                string strName = name;

                std::println("============================================");
                std::println("KERNEL: \"{}\"", strName);
                int value;

                cuKernelGetAttribute(&value, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel, cuDevice);
                std::println("    registers per thread  {:10}", value);

                cuKernelGetAttribute(&value, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel,
                                     cuDevice);
                std::println("    max threads per block {:10}", value);

                cuKernelGetAttribute(&value, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel, cuDevice);
                std::println("    shared memory         {:10}", value);

                cuKernelGetAttribute(&value, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, kernel, cuDevice);
                std::println("    constant memory       {:10}", value);

                cuKernelGetAttribute(&value, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel, cuDevice);
                std::println("    local memory          {:10}", value);

                cuKernelGetAttribute(&value, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                     kernel, cuDevice);
                std::println("    max dynamic memory    {:10}", value);

                kernelNames.push_back(strName);
                kernels[strName] = function;
            }
        }

        // for(string kernelName : kernelNames){
        // 	CUfunction kernel;
        // 	cu_checked(cuModuleGetFunction(&kernel, mod, kernelName.c_str()));

        // 	kernels[kernelName] = kernel;
        // }

        for (auto &callback : compileCallbacks) {
            callback();
        }

        printElapsedTime("link duration: ", tStart);
    }

    void onCompile(std::function<void(void)> callback) { compileCallbacks.push_back(callback); }

    void launch(string kernelName, void *args[], OptionalLaunchSettings launchArgs = {}) {

        auto res_launch = cuLaunchKernel(kernels[kernelName], launchArgs.gridsize, 1, 1,
                                         launchArgs.blocksize, 1, 1, 0, 0, args, nullptr);

        if (res_launch != CUDA_SUCCESS) {
            const char *str;
            cuGetErrorString(res_launch, &str);
            printf("error: %s \n", str);
            std::cout << __FILE__ << " - " << __LINE__ << std::endl;
        }
    }

    void launch(string kernelName, void *args[], int count) {

        uint32_t blockSize = 256;
        uint32_t gridSize = (count + blockSize - 1) / blockSize;

        auto res_launch = cuLaunchKernel(kernels[kernelName], gridSize, 1, 1, blockSize, 1, 1, 0, 0,
                                         args, nullptr);

        if (res_launch != CUDA_SUCCESS) {
            const char *str;
            cuGetErrorString(res_launch, &str);
            printf("error: %s \n", str);
            std::cout << __FILE__ << " - " << __LINE__ << std::endl;
        }
    }

    int maxOccupancy(string kernelName, int blockSize) {
        CUdevice device;
        int numSMs;
        cuCtxGetDevice(&device);
        cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

        int numBlocks;
        CUresult resultcode = cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks, kernels[kernelName], blockSize, 0);
        numBlocks *= numSMs;

        // numGroups = 100;
        //  make sure at least 10 workgroups are spawned)
        numBlocks = std::clamp(numBlocks, 10, 100'000);

        return numBlocks;
    }

    void launchCooperative(string kernelName, void *args[],
                           OptionalLaunchSettings launchArgs = {}) {

        CUevent event_start = events_launch_start[kernelName];
        CUevent event_end = events_launch_end[kernelName];

        // cuEventRecord(event_start, 0);
        int blockSize = launchArgs.blocksize > 0 ? launchArgs.blocksize : 128;

        int numBlocks;
        auto kernel = this->kernels[kernelName];
        auto res_launch = cuLaunchCooperativeKernel(kernel, maxOccupancy(kernelName, blockSize), 1,
                                                    1, blockSize, 1, 1, 0, 0, args);

        if (res_launch != CUDA_SUCCESS) {
            const char *str;
            cuGetErrorString(res_launch, &str);
            printf("error: %s \n", str);
            std::cout << __FILE__ << " - " << __LINE__ << std::endl;
        }

        // cuEventRecord(event_end, 0);

        // if(launchArgs.measureDuration){
        // 	cuCtxSynchronize();

        // 	float duration;
        // 	cuEventElapsedTime(&duration, event_start, event_end);

        // 	last_launch_duration[kernelName] = duration;
        // }
    }
};