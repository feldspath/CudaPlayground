

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
#include "OrbitControls.h"
#include "cudaGL.h"
// #include "builtin_types.h"

#include "ObjLoader.h"
#include "unsuck.hpp"

#include "HostDeviceInterface.h"

CUdeviceptr cptr_buffer;
CUdeviceptr cptr_positions, cptr_uvs, cptr_colors;
CUdeviceptr cptr_texture;

CUgraphicsResource cugl_colorbuffer;
CudaModularProgram *cuda_program = nullptr;
CUevent cevent_start, cevent_end;

std::shared_ptr<ObjData> model;
// vector<uint32_t> colors;

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
    uniforms.sampleMode = sampleMode;

    glm::mat4 rotX = glm::rotate(glm::mat4(), 3.1415f * 0.5f, glm::vec3(1.0, 0.0, 0.0));

    glm::mat4 world = rotX;
    glm::mat4 view = renderer->camera->viewMatrix();
    glm::mat4 proj = renderer->camera->projMatrix();
    glm::mat4 worldViewProj = proj * view * world;
    world = glm::transpose(world);
    view = glm::transpose(view);
    proj = glm::transpose(proj);
    worldViewProj = glm::transpose(worldViewProj);
    memcpy(&uniforms.world, &world, sizeof(world));
    memcpy(&uniforms.view, &view, sizeof(view));
    memcpy(&uniforms.proj, &proj, sizeof(proj));
    memcpy(&uniforms.transform, &worldViewProj, sizeof(worldViewProj));

    float values[16];
    memcpy(&values, &worldViewProj, sizeof(worldViewProj));

    void *args[] = {&uniforms,       &cptr_buffer, &output_surf, &model->numTriangles,
                    &cptr_positions, &cptr_uvs,    &cptr_colors, &cptr_texture};

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

void initCudaProgram(std::shared_ptr<GLRenderer> renderer, std::shared_ptr<ObjData> model,
                     std::vector<uint32_t> &texture) {

    cuMemAlloc(&cptr_buffer, 100'000'000);

    int numVertices = model->numTriangles * 3;
    cuMemAlloc(&cptr_positions, numVertices * 12);
    cuMemAlloc(&cptr_uvs, numVertices * 8);
    cuMemcpyHtoD(cptr_positions, model->xyz.data(), numVertices * 12);
    cuMemcpyHtoD(cptr_uvs, model->uv.data(), numVertices * 8);

    cuMemAlloc(&cptr_texture, 4 * 1024 * 1024);
    cuMemcpyHtoD(cptr_texture, texture.data(), 4 * 1024 * 1024);

    cuda_program = new CudaModularProgram({.modules =
                                               {
                                                   "./modules/rasterizeTriangles/rasterize.cu",
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

    auto renderer = std::make_shared<GLRenderer>(std::make_shared<Camera3D>(),
                                                 std::make_shared<OrbitControls>());

    OrbitControls &controls = *dynamic_pointer_cast<OrbitControls>(renderer->controls);

    controls.yaw = -2.6;
    controls.pitch = -0.4;
    controls.radius = 6.0;
    controls.target = {0.0f, 0.0f, 0.0f};

    initCuda();

    model = ObjLoader::load("./resources/spot/spot_triangulated.obj");
    auto ppmdata = readBinaryFile("./resources/spot/spot_texture.ppm", 40, 1000000000000);
    std::vector<uint32_t> colors(1024 * 1024, 0);

    for (int i = 0; i < 1024 * 1024; i++) {
        uint32_t r = ppmdata->get<uint8_t>(3 * i + 0);
        uint32_t g = ppmdata->get<uint8_t>(3 * i + 1);
        uint32_t b = ppmdata->get<uint8_t>(3 * i + 2);
        uint32_t color = r | (g << 8) | (b << 16);

        colors[i] = color;
    }

    initCudaProgram(renderer, model, colors);

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

            ImGui::BulletText("Cuda software rasterizer rendering 25 instances of the spot model "
                              "\n(5856 triangles, each).");
            ImGui::BulletText("Each cuda block renders one triangle, \nwith each thread processing "
                              "a different fragment.");
            ImGui::BulletText("Cuda Kernel: rasterizeTriangles/rasterize.cu");
            ImGui::BulletText("Spot model courtesy of Keenan Crane.");

            ImGui::End();
        }

        { // SETTINGS WINDOW

            ImGui::SetNextWindowPos(ImVec2(10, 280 + 180 + 10));
            ImGui::SetNextWindowSize(ImVec2(490, 230));

            ImGui::Begin("Settings");

            ImGui::Text("Color:");
            ImGui::RadioButton("Texture", &colorMode, COLORMODE_TEXTURE);
            ImGui::RadioButton("UVs", &colorMode, COLORMODE_UV);
            ImGui::RadioButton("Triangle Index", &colorMode, COLORMODE_TRIANGLE_ID);
            ImGui::RadioButton("Time", &colorMode, COLORMODE_TIME);
            ImGui::RadioButton("Time (normalized)", &colorMode, COLORMODE_TIME_NORMALIZED);

            ImGui::Text("Sampling:");
            ImGui::RadioButton("Nearest", &sampleMode, SAMPLEMODE_NEAREST);
            ImGui::RadioButton("Linear", &sampleMode, SAMPLEMODE_LINEAR);

            ImGui::End();
        }
    };

    renderer->loop(update, render);

    return 0;
}
