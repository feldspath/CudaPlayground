

#include <filesystem>
#include <format>
#include <iostream>
#include <locale.h>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "Controls2D.h"
#include "CudaModularProgram.h"
#include "GLRenderer.h"
#include "ObjLoader.h"
#include "cudaGL.h"
#include "lodepng.h"
#include "unsuck.hpp"

#include "HostDeviceInterface.h"

CUdeviceptr cptr_buffer;
CUdeviceptr cptr_grid;
CUdeviceptr cptr_entities;
CUdeviceptr cptr_gameState;
CUdeviceptr cptr_spriteSheet;
CUdeviceptr cptr_ascii_16;
uint32_t gridRows;
uint32_t gridCols;

CUgraphicsResource cugl_colorbuffer;
CudaModularProgram *cuda_program = nullptr;
CudaModularProgram *cuda_update = nullptr;
CUevent cevent_start, cevent_end;

int renderMode = RENDERMODE_DEFAULT;
bool printTimings = false;
bool creativeMode = false;

void initCuda() {
    cuInit(0);
    CUdevice cuDevice;
    CUcontext context;
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&context, 0, cuDevice);
}

int maxOccupancy(CudaModularProgram *program, const char *kernel, int workgroupSize, int numSMs) {
    int numGroups;
    int resultcode = cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &numGroups, program->kernels[kernel], workgroupSize, 0);
    numGroups *= numSMs;

    // numGroups = 100;
    //  make sure at least 10 workgroups are spawned)
    numGroups = std::clamp(numGroups, 10, 100'000);
    return numGroups;
}

void updateCUDA(std::shared_ptr<GLRenderer> renderer) {
    CUdevice device;
    int numSMs;
    cuCtxGetDevice(&device);
    cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

    int workgroupSize = 64;

    int numGroups = maxOccupancy(cuda_update, "update", workgroupSize, numSMs);

    cuEventRecord(cevent_start, 0);

    float time = now();

    auto runtime = Runtime::getInstance();

    Uniforms uniforms;
    uniforms.width = renderer->width;
    uniforms.height = renderer->height;
    uniforms.time = now();
    uniforms.renderMode = renderMode;
    uniforms.modeId = runtime->modeId;
    uniforms.printTimings = printTimings;
    uniforms.creativeMode = creativeMode;

    memcpy(&uniforms.cursorPos, &runtime->mousePosition, sizeof(runtime->mousePosition));
    uniforms.mouseButtons = Runtime::getInstance()->mouseButtons;

    glm::mat4 view = renderer->camera->viewMatrix();
    glm::mat4 proj = renderer->camera->projMatrix();
    // glm::mat4 proj = glm::ortho(0.0, 100.0, 0.0, 100.0 / renderer->camera->aspect);
    glm::mat4 invproj = glm::inverse(proj);
    glm::mat4 invview = glm::inverse(view);
    view = glm::transpose(view);
    proj = glm::transpose(proj);
    invproj = glm::transpose(invproj);
    invview = glm::transpose(invview);
    memcpy(&uniforms.view, &view, sizeof(view));
    memcpy(&uniforms.proj, &proj, sizeof(proj));
    memcpy(&uniforms.invproj, &invproj, sizeof(invproj));
    memcpy(&uniforms.invview, &invview, sizeof(invview));

    void *args[] = {&uniforms, &cptr_gameState, &cptr_buffer,  &gridRows,
                    &gridCols, &cptr_grid,      &cptr_entities};

    auto res_launch = cuLaunchCooperativeKernel(cuda_update->kernels["update"], numGroups, 1, 1,
                                                workgroupSize, 1, 1, 0, 0, args);

    if (res_launch != CUDA_SUCCESS) {
        const char *str;
        cuGetErrorString(res_launch, &str);
        printf("error: %s \n", str);
    }

    // cuEventRecord(cevent_end, 0);
    // cuEventSynchronize(cevent_end);

    //{
    //    float total_ms;
    //    cuEventElapsedTime(&total_ms, cevent_start, cevent_end);

    //    std::cout << "Update duration: " << std::format("total:     {:6.1f} ms\n", total_ms);
    //}

    cuCtxSynchronize();
}

void renderCUDA(std::shared_ptr<GLRenderer> renderer) {
    cuGraphicsGLRegisterImage(&cugl_colorbuffer,
                              renderer->view.framebuffer->colorAttachments[0]->handle,
                              GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

    CUresult resultcode = CUDA_SUCCESS;

    CUdevice device;
    int numSMs;
    cuCtxGetDevice(&device);
    cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

    int workgroupSize = 64;

    int numGroups = maxOccupancy(cuda_program, "kernel", workgroupSize, numSMs);

    std::vector<CUgraphicsResource> dynamic_resources = {cugl_colorbuffer};
    cuGraphicsMapResources(dynamic_resources.size(), dynamic_resources.data(),
                           ((CUstream)CU_STREAM_DEFAULT));
    CUDA_RESOURCE_DESC res_desc = {};
    res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
    cuGraphicsSubResourceGetMappedArray(&res_desc.res.array.hArray, cugl_colorbuffer, 0, 0);
    CUsurfObject output_surf;
    cuSurfObjectCreate(&output_surf, &res_desc);

    cuEventRecord(cevent_start, 0);

    float time = now();

    Uniforms uniforms;
    uniforms.width = renderer->width;
    uniforms.height = renderer->height;
    uniforms.time = now();
    uniforms.renderMode = renderMode;
    uniforms.printTimings = printTimings;
    uniforms.creativeMode = creativeMode;

    glm::mat4 view = renderer->camera->viewMatrix();
    glm::mat4 proj = renderer->camera->projMatrix();
    // glm::mat4 proj = glm::ortho(0.0, 100.0, 0.0, 100.0 / renderer->camera->aspect);
    glm::mat4 invproj = glm::inverse(proj);
    glm::mat4 invview = glm::inverse(view);
    view = glm::transpose(view);
    proj = glm::transpose(proj);
    invproj = glm::transpose(invproj);
    invview = glm::transpose(invview);
    memcpy(&uniforms.view, &view, sizeof(view));
    memcpy(&uniforms.proj, &proj, sizeof(proj));
    memcpy(&uniforms.invproj, &invproj, sizeof(invproj));
    memcpy(&uniforms.invview, &invview, sizeof(invview));

    void *args[] = {&uniforms, &cptr_gameState, &cptr_buffer,   &output_surf,   &gridRows,
                    &gridCols, &cptr_grid,      &cptr_entities, &cptr_ascii_16, &cptr_spriteSheet};

    auto res_launch = cuLaunchCooperativeKernel(cuda_program->kernels["kernel"], numGroups, 1, 1,
                                                workgroupSize, 1, 1, 0, 0, args);

    if (res_launch != CUDA_SUCCESS) {
        const char *str;
        cuGetErrorString(res_launch, &str);
        printf("error: %s \n", str);
    }

    // cuEventRecord(cevent_end, 0);
    // cuEventSynchronize(cevent_end);

    //{
    //    float total_ms;
    //    cuEventElapsedTime(&total_ms, cevent_start, cevent_end);

    //    std::cout << "Render duration: " << std::format("total:     {:6.1f} ms\n", total_ms);
    //}

    cuCtxSynchronize();

    cuSurfObjectDestroy(output_surf);
    cuGraphicsUnmapResources(dynamic_resources.size(), dynamic_resources.data(),
                             ((CUstream)CU_STREAM_DEFAULT));
    cuGraphicsUnregisterResource(cugl_colorbuffer);
}

void initGameState() {
    GameState state;
    state.playerMoney = 2000;

    cuMemcpyHtoD(cptr_gameState, &state, sizeof(GameState));
}

void initCudaProgram(std::shared_ptr<GLRenderer> renderer, std::vector<uint8_t> &img_ascii_16,
                     std::vector<uint8_t> &img_spritesheet) {
    cuMemAlloc(&cptr_buffer, 100'000'000);
    cuMemAlloc(&cptr_gameState, sizeof(GameState));

    initGameState();

    gridRows = 512;
    gridCols = 512;
    int numCells = gridRows * gridCols;
    cuMemAlloc(&cptr_grid, numCells * BYTES_PER_CELL);

    std::vector<char> gridCells(numCells * BYTES_PER_CELL);
    for (int y = 0; y < gridRows; ++y) {
        for (int x = 0; x < gridCols; ++x) {
            int cellId = y * gridCols + x;
            *(reinterpret_cast<int32_t *>(gridCells.data() + cellId * BYTES_PER_CELL)) = GRASS;
        }
    }
    cuMemcpyHtoD(cptr_grid, gridCells.data(), gridCells.size());

    // Let's assume we can have as much entities as we have cells
    cuMemAlloc(&cptr_entities, sizeof(uint32_t) + numCells * (BYTES_PER_ENTITY));
    uint32_t entitiesCount = 0;
    cuMemcpyHtoD(cptr_entities, &entitiesCount, sizeof(uint32_t));

    // Font rendering
    cuMemAlloc(&cptr_ascii_16, img_ascii_16.size());
    cuMemcpyHtoD(cptr_ascii_16, img_ascii_16.data(), img_ascii_16.size());

    // Sprites
    cuMemAlloc(&cptr_spriteSheet, img_spritesheet.size());
    cuMemcpyHtoD(cptr_spriteSheet, img_spritesheet.data(), img_spritesheet.size());

    cuda_program = new CudaModularProgram({.modules =
                                               {
                                                   "./modules/shapes2D/rasterize.cu",
                                                   "./modules/common/utils.cu",
                                                   "./modules/shapes2D/path.cu",
                                               },
                                           .kernels = {"kernel"}});

    cuda_update = new CudaModularProgram({.modules =
                                              {
                                                  "./modules/common/utils.cu",
                                                  "./modules/shapes2D/update.cu",
                                                  "./modules/shapes2D/path.cu",
                                                  "./modules/shapes2D/pathfinding.cu",
                                              },
                                          .kernels = {"update"}});

    cuEventCreate(&cevent_start, 0);
    cuEventCreate(&cevent_end, 0);

    cuGraphicsGLRegisterImage(&cugl_colorbuffer,
                              renderer->view.framebuffer->colorAttachments[0]->handle,
                              GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
}

int main() {

    std::cout << std::setprecision(2) << std::fixed;
    setlocale(LC_ALL, "en_AT.UTF-8");

    auto camera = std::make_shared<Camera2D>();
    auto controls = std::make_shared<Controls2D>(camera);

    auto renderer = std::make_shared<GLRenderer>(camera, controls);

    initCuda();

    std::vector<uint8_t> img_ascii_16;
    {
        auto asciidata = readBinaryFile("./resources/tilesets/ascii_16.png");
        uint32_t width;
        uint32_t height;
        lodepng::decode(img_ascii_16, width, height, (const unsigned char *)asciidata->data_char,
                        size_t(asciidata->size));
    }

    std::vector<uint8_t> img_spritesheet;
    {
        auto data = readBinaryFile("./resources/sprites/sprite_sheet.png");
        uint32_t width;
        uint32_t height;
        lodepng::decode(img_spritesheet, width, height, (const unsigned char *)data->data_char,
                        size_t(data->size));
    }

    initCudaProgram(renderer, img_ascii_16, img_spritesheet);

    auto update = [&]() { updateCUDA(renderer); };

    auto render = [&]() {
        renderer->view.framebuffer->setSize(renderer->width, renderer->height);

        glBindFramebuffer(GL_FRAMEBUFFER, renderer->view.framebuffer->handle);

        renderCUDA(renderer);

        { // INFO WINDOW

            ImGui::SetNextWindowPos(ImVec2(10, 280));
            ImGui::SetNextWindowSize(ImVec2(490, 180));

            ImGui::Begin("Infos");

            ImGui::BulletText("Cuda Render Kernel: shapes2D/rasterize.cu");
            ImGui::BulletText("Cuda Update Kernel: shapes2D/update.cu");

            ImGui::End();
        }

        { // SETTINGS WINDOW

            ImGui::SetNextWindowPos(ImVec2(10, 280 + 180 + 10));
            ImGui::SetNextWindowSize(ImVec2(490, 230));

            ImGui::Begin("Settings");

            ImGui::Text("Render mode:");
            ImGui::RadioButton("Default", &renderMode, RENDERMODE_DEFAULT);
            ImGui::RadioButton("Network", &renderMode, RENDERMODE_NETWORK);

            ImGui::Text("Options");
            ImGui::Checkbox("Pring Timings", &printTimings);
            ImGui::Checkbox("Creative Mode", &creativeMode);

            ImGui::End();
        }
    };

    renderer->loop(update, render);

    return 0;
}
