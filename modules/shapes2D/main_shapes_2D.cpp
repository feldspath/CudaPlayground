

#include <filesystem>
#include <format>
#include <iostream>
#include <locale.h>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "CudaModularProgram.h"
#include "GLRenderer.h"
#include "cudaGL.h"
// #include "builtin_types.h"

#include "Controls2D.h"
#include "ObjLoader.h"
#include "unsuck.hpp"

#include "HostDeviceInterface.h"

CUdeviceptr cptr_buffer;
CUdeviceptr cptr_grid;
uint32_t gridRows;
uint32_t gridCols;

CUgraphicsResource cugl_colorbuffer;
CudaModularProgram *cuda_program = nullptr;
CUevent cevent_start, cevent_end;

int colorMode = COLORMODE_TEXTURE;
int sampleMode = SAMPLEMODE_LINEAR;

void initCuda() {
    cuInit(0);
    CUdevice cuDevice;
    CUcontext context;
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&context, 0, cuDevice);
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

    int workgroupSize = 128;

    int numGroups;
    resultcode = cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &numGroups, cuda_program->kernels["kernel"], workgroupSize, 0);
    numGroups *= numSMs;

    // numGroups = 100;
    //  make sure at least 10 workgroups are spawned)
    numGroups = std::clamp(numGroups, 10, 100'000);

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
    uniforms.colorMode = colorMode;

    glm::mat4 rotX = glm::rotate(glm::mat4(), 3.1415f * 0.5f, glm::vec3(1.0, 0.0, 0.0));

    glm::mat4 world = rotX;
    glm::mat4 view = renderer->camera->viewMatrix();
    glm::mat4 proj = renderer->camera->projMatrix();
    // glm::mat4 proj = glm::ortho(0.0, 100.0, 0.0, 100.0 / renderer->camera->aspect);
    glm::mat4 invproj = glm::inverse(proj);
    glm::mat4 invview = glm::inverse(view);
    glm::mat4 worldViewProj = proj * view * world;
    world = glm::transpose(world);
    view = glm::transpose(view);
    proj = glm::transpose(proj);
    invproj = glm::transpose(invproj);
    invview = glm::transpose(invview);
    worldViewProj = glm::transpose(worldViewProj);
    memcpy(&uniforms.world, &world, sizeof(world));
    memcpy(&uniforms.view, &view, sizeof(view));
    memcpy(&uniforms.proj, &proj, sizeof(proj));
    memcpy(&uniforms.invproj, &invproj, sizeof(invproj));
    memcpy(&uniforms.invview, &invview, sizeof(invview));
    memcpy(&uniforms.transform, &worldViewProj, sizeof(worldViewProj));

    float values[16];
    memcpy(&values, &worldViewProj, sizeof(worldViewProj));

    void *args[] = {&uniforms, &cptr_buffer, &output_surf, &gridRows, &gridCols, &cptr_grid};

    auto res_launch = cuLaunchCooperativeKernel(cuda_program->kernels["kernel"], numGroups, 1, 1,
                                                workgroupSize, 1, 1, 0, 0, args);

    if (res_launch != CUDA_SUCCESS) {
        const char *str;
        cuGetErrorString(res_launch, &str);
        printf("error: %s \n", str);
    }

    cuEventRecord(cevent_end, 0);
    // cuEventSynchronize(cevent_end);

    // {
    // 	float total_ms;
    // 	cuEventElapsedTime(&total_ms, cevent_start, cevent_end);

    // 	cout << "CUDA durations: " << endl;
    // 	cout << std::format("total:     {:6.1f} ms", total_ms) << endl;
    // }

    cuCtxSynchronize();

    cuSurfObjectDestroy(output_surf);
    cuGraphicsUnmapResources(dynamic_resources.size(), dynamic_resources.data(),
                             ((CUstream)CU_STREAM_DEFAULT));
    cuGraphicsUnregisterResource(cugl_colorbuffer);
}

void initCudaProgram(std::shared_ptr<GLRenderer> renderer) {
    cuMemAlloc(&cptr_buffer, 100'000'000);

    gridRows = 20;
    gridCols = 20;
    int numCells = gridRows * gridCols;
    cuMemAlloc(&cptr_grid, numCells * sizeof(uint32_t));
    std::vector<uint32_t> gridCells(numCells);
    for (int y = 0; y < gridRows; ++y) {
        for (int x = 0; x < gridCols; ++x) {
            // float3 color;
            // if ((i + j) % 2 == 0) {
            //     color = float3{0.0f, 1.0f, 1.0f};
            // } else {
            //     color = float3{1.0f, 1.0f, 0.0f};
            // }
            // voxels->colors[i * 10 + j] = color;
            gridCells[y * gridCols + x] = 0;
        }
    }

    gridCells[gridCols * 5 + 2] = -1;

    cuMemcpyHtoD(cptr_grid, gridCells.data(), gridCells.size() * sizeof(uint32_t));

    cuda_program = new CudaModularProgram({.modules =
                                               {
                                                   "./modules/shapes2D/rasterize.cu",
                                                   "./modules/common/utils.cu",
                                               },
                                           .kernels = {"kernel"}});

    cuEventCreate(&cevent_start, 0);
    cuEventCreate(&cevent_end, 0);

    cuGraphicsGLRegisterImage(&cugl_colorbuffer,
                              renderer->view.framebuffer->colorAttachments[0]->handle,
                              GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
}

int main() {

    std::cout << std::setprecision(2) << std::fixed;
    setlocale(LC_ALL, "en_AT.UTF-8");

    auto renderer =
        std::make_shared<GLRenderer>(std::make_shared<Camera2D>(), std::make_shared<Controls2D>());

    initCuda();

    initCudaProgram(renderer);

    auto update = [&]() {

    };

    auto render = [&]() {
        renderer->view.framebuffer->setSize(renderer->width, renderer->height);

        glBindFramebuffer(GL_FRAMEBUFFER, renderer->view.framebuffer->handle);

        renderCUDA(renderer);

        { // INFO WINDOW

            ImGui::SetNextWindowPos(ImVec2(10, 280));
            ImGui::SetNextWindowSize(ImVec2(490, 180));

            ImGui::Begin("Infos");

            ImGui::BulletText("Cuda Kernel: rasterizeVoxels/rasterize.cu");
            ImGui::BulletText("Spot model courtesy of Keenan Crane.");

            ImGui::End();
        }

        { // SETTINGS WINDOW

            ImGui::SetNextWindowPos(ImVec2(10, 280 + 180 + 10));
            ImGui::SetNextWindowSize(ImVec2(490, 230));

            ImGui::Begin("Settings");

            ImGui::Text("Color:");
            ImGui::RadioButton("Triangle Index", &colorMode, COLORMODE_ID);
            ImGui::RadioButton("Time", &colorMode, COLORMODE_TIME);
            ImGui::RadioButton("Time (normalized)", &colorMode, COLORMODE_TIME_NORMALIZED);

            ImGui::End();
        }
    };

    renderer->loop(update, render);

    return 0;
}
