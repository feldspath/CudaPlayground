

#include <filesystem>
#include <format>
#include <iostream>
#include <locale.h>
#include <memory>
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

using std::shared_ptr;

namespace fs = std::filesystem;

CUdeviceptr cptr_buffer;
CUdeviceptr cptr_chunk;
CUdeviceptr cptr_entities;
CUdeviceptr cptr_savedFields;
CUdeviceptr cptr_gameState;
CUdeviceptr cptr_spriteSheet;
CUdeviceptr cptr_ascii_32;
CUdeviceptr cptr_graphNetwork;
uint32_t chunksRows;
uint32_t chunksCols;

CUgraphicsResource cugl_colorbuffer;
CudaModularProgram *cuda_program = nullptr;
CudaModularProgram *cuda_update = nullptr;
CUevent cevent_start, cevent_end;

int renderMode = RENDERMODE_DEFAULT;
bool printTimings = false;
bool creativeMode = false;
bool displayFlowfield = false;
float timeMultiplier = 1.0f;
unsigned int blockSize = 128;

void initCuda() {
    cuInit(0);
    CUdevice cuDevice;
    CUcontext context;
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&context, 0, cuDevice);
}

void saveMap() {

    // assume map size is fixed, for now, so we dont store it in file
    int numChunks = chunksRows * chunksCols;

    Buffer chunks(numChunks * sizeof(Chunk));
    cuMemcpyDtoH(chunks.data, cptr_chunk, chunks.size);

    int compressedChunkSize = sizeof(Cell) * CHUNK_SIZE;
    Buffer compressedChunks(numChunks * compressedChunkSize);
    for (int i = 0; i < numChunks; i++) {
        memcpy(compressedChunks.data_u8 + i * compressedChunkSize,
               chunks.data_u8 + sizeof(Chunk) * i, compressedChunkSize);
    }

    Buffer entities(2 * sizeof(uint32_t) + MAX_ENTITY_COUNT * sizeof(Entity));
    cuMemcpyDtoH(entities.data, cptr_entities, entities.size);

    GameState state;
    cuMemcpyDtoH(&state, cptr_gameState, sizeof(state));

    Buffer buffer(compressedChunks.size + entities.size + sizeof(state));

    int64_t offsetGridCells = 0;
    int64_t offsetEntities = offsetGridCells + compressedChunks.size;
    int64_t offsetState = offsetEntities + entities.size;

    memcpy(buffer.data_u8 + offsetGridCells, compressedChunks.data, compressedChunks.size);
    memcpy(buffer.data_u8 + offsetEntities, entities.data, entities.size);
    memcpy(buffer.data_u8 + offsetState, &state, sizeof(state));

    writeBinaryFile("./savefile.save", buffer);
}

void loadMap() {

    if (!fs::exists("./savefile.save"))
        return;

    shared_ptr<Buffer> buffer = readBinaryFile("./savefile.save");

    // assume map size is fixed, for now, so we dont store it in file
    int numChunks = chunksRows * chunksCols;

    int compressedChunkSize = sizeof(Cell) * CHUNK_SIZE;
    Buffer compressedChunks(numChunks * compressedChunkSize);
    Buffer entities(2 * sizeof(uint32_t) + MAX_ENTITY_COUNT * sizeof(Entity));
    GameState state;

    int64_t offsetGridCells = 0;
    int64_t offsetEntities = offsetGridCells + compressedChunks.size;
    int64_t offsetState = offsetEntities + entities.size;

    memcpy(compressedChunks.data, buffer->data_u8 + offsetGridCells, compressedChunks.size);
    memcpy(entities.data, buffer->data_u8 + offsetEntities, entities.size);
    memcpy(&state, buffer->data_u8 + offsetState, sizeof(state));

    Buffer chunks(numChunks * sizeof(Chunk));
    for (unsigned int i = 0; i < numChunks; i++) {
        memcpy(chunks.data_u8 + i * sizeof(Chunk),
               compressedChunks.data_u8 + compressedChunkSize * i, compressedChunkSize);
        ((Chunk *)chunks.data)[i].offset = int2(i % chunksCols, i / chunksCols);
    }

    state.firstFrame = true;
    state.gameTime.dt = 0.0f;

    // memset(entities.data, 0, entities.size);

    cuMemcpyHtoD(cptr_chunk, chunks.data, chunks.size);
    cuMemcpyHtoD(cptr_entities, entities.data, entities.size);
    cuMemcpyHtoD(cptr_gameState, &state, sizeof(GameState));
}

void updateCUDA(std::shared_ptr<GLRenderer> renderer) {
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
    uniforms.timeMultiplier = timeMultiplier;
    uniforms.displayFlowfield = displayFlowfield;

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

    GameData gamedata;
    gamedata.uniforms = uniforms;
    gamedata.state = (GameState *)cptr_gameState;
    gamedata.buffer = (unsigned int *)cptr_buffer;
    gamedata.numRows = chunksRows;
    gamedata.numCols = chunksCols;
    gamedata.chunks = (Chunk *)cptr_chunk;
    gamedata.entitiesBuffer = (void *)cptr_entities;
    gamedata.savedFieldsBuffer = (void *)cptr_savedFields;
    gamedata.graphNetworkBuffer = (NetworkNode *)cptr_graphNetwork;
    gamedata.img_ascii_16 = (uint32_t *)cptr_ascii_32;
    gamedata.img_spritesheet = (uint32_t *)cptr_spriteSheet;

    void *args[] = {&gamedata};

    cuda_update->launchCooperative("update", args, {.blocksize = blockSize});

    cuCtxSynchronize();
}

void renderCUDA(std::shared_ptr<GLRenderer> renderer) {
    cuGraphicsGLRegisterImage(&cugl_colorbuffer,
                              renderer->view.framebuffer->colorAttachments[0]->handle,
                              GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

    CUresult resultcode = CUDA_SUCCESS;

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

    auto runtime = Runtime::getInstance();

    Uniforms uniforms;
    uniforms.width = renderer->width;
    uniforms.height = renderer->height;
    uniforms.time = now();
    uniforms.renderMode = renderMode;
    uniforms.printTimings = printTimings;
    uniforms.creativeMode = creativeMode;
    uniforms.timeMultiplier = timeMultiplier;
    uniforms.displayFlowfield = displayFlowfield;

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

    GameData gamedata;
    gamedata.uniforms = uniforms;
    gamedata.state = (GameState *)cptr_gameState;
    gamedata.buffer = (unsigned int *)cptr_buffer;
    gamedata.numRows = chunksRows;
    gamedata.numCols = chunksCols;
    gamedata.chunks = (Chunk *)cptr_chunk;
    gamedata.entitiesBuffer = (void *)cptr_entities;
    gamedata.graphNetworkBuffer = (NetworkNode *)cptr_graphNetwork;
    gamedata.img_ascii_16 = (uint32_t *)cptr_ascii_32;
    gamedata.img_spritesheet = (uint32_t *)cptr_spriteSheet;

    void *args[] = {&gamedata, &output_surf};

    cuda_program->launchCooperative("kernel", args, {.blocksize = blockSize});

    cuCtxSynchronize();

    cuSurfObjectDestroy(output_surf);
    cuGraphicsUnmapResources(dynamic_resources.size(), dynamic_resources.data(),
                             ((CUstream)CU_STREAM_DEFAULT));
    cuGraphicsUnregisterResource(cugl_colorbuffer);
}

void initGameState() {
    GameState state;
    state.firstFrame = true;
    state.playerMoney = 2000;
    state.buildingDisplay = MapId::invalidId();
    state.gameTime = GameTime();

    cuMemcpyHtoD(cptr_gameState, &state, sizeof(GameState));
}

void initCudaProgram(std::shared_ptr<GLRenderer> renderer, std::vector<uint8_t> &img_ascii_32,
                     std::vector<uint8_t> &img_spritesheet) {
    // kernel
    cuda_program =
        new CudaModularProgram({.modules =
                                    {
                                        "./modules/common/utils.cu",
                                        "./modules/shapes2D/Rendering/rasterize.cu",
                                        "./modules/shapes2D/Rendering/gui.cu",
                                        "./modules/shapes2D/Rendering/sprite.cu",
                                        "./modules/shapes2D/World/time.cu",
                                        "./modules/shapes2D/World/Path/path.cu",
                                        "./modules/shapes2D/World/direction.cu",
                                    },
                                .customIncludeDirs = {"./modules/shapes2D", " ./modules"}});

    cuda_update =
        new CudaModularProgram({.modules =
                                    {
                                        "./modules/common/utils.cu",
                                        "./modules/shapes2D/World/update.cu",
                                        "./modules/shapes2D/World/Path/path.cu",
                                        "./modules/shapes2D/World/Path/pathfinding.cu",
                                        "./modules/shapes2D/World/time.cu",
                                        "./modules/shapes2D/World/Entities/entities.cu",
                                        "./modules/shapes2D/World/Entities/movement.cu",
                                        "./modules/shapes2D/World/direction.cu",
                                    },
                                .customIncludeDirs = {"./modules/shapes2D", " ./modules"}});

    // Buffers
    cuMemAlloc(&cptr_buffer, 400'000'000);
    cuMemAlloc(&cptr_gameState, sizeof(GameState));

    initGameState();

    // Allocate the chunks
    chunksCols = N_CHUNK_X;
    chunksRows = N_CHUNK_Y;
    int maxOccupancy = cuda_program->maxOccupancy("kernel", blockSize);
    cuMemAlloc(&cptr_chunk, sizeof(Chunk) * chunksRows * chunksCols);
    cuMemAlloc(&cptr_savedFields, sizeof(IntegrationField) * maxOccupancy);

    std::vector<Chunk> chunks(chunksRows * chunksCols);

    for (int j = 0; j < chunksRows; j++) {
        for (int i = 0; i < chunksCols; i++) {
            auto &chunk = chunks[j * chunksCols + i];
            chunk.offset = {i, j};
            // std::cout << "offset:" << i << "," << j << "\n";
            for (int y = 0; y < CHUNK_X; ++y) {
                for (int x = 0; x < CHUNK_Y; ++x) {
                    int cellId = y * CHUNK_X + x;
                    chunk.cells[cellId].cell.tileId = ROAD;
                    chunk.cachedFlowfields[cellId].state = INVALID;
                }
            }
        }
    }

    std::vector<IntegrationField> savedFields(maxOccupancy);
    for (auto &f : savedFields) {
        f.ongoingComputation = false;
    }

    cuMemcpyHtoD(cptr_chunk, chunks.data(), sizeof(Chunk) * chunks.size());
    cuMemcpyHtoD(cptr_savedFields, savedFields.data(),
                 savedFields.size() * sizeof(IntegrationField));

    // Graph
    cuMemAlloc(&cptr_graphNetwork, 2 * sizeof(uint32_t) + MAX_NETWORK_NODES * sizeof(NetworkNode));

    // Entities
    cuMemAlloc(&cptr_entities, 2 * sizeof(uint32_t) + MAX_ENTITY_COUNT * sizeof(Entity));
    uint32_t init[2] = {0, 0};
    cuMemcpyHtoD(cptr_entities, init, 2 * sizeof(uint32_t));

    // Font rendering
    cuMemAlloc(&cptr_ascii_32, img_ascii_32.size());
    cuMemcpyHtoD(cptr_ascii_32, img_ascii_32.data(), img_ascii_32.size());

    // Sprites
    cuMemAlloc(&cptr_spriteSheet, img_spritesheet.size());
    cuMemcpyHtoD(cptr_spriteSheet, img_spritesheet.data(), img_spritesheet.size());

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

    std::vector<uint8_t> img_ascii;
    {
        auto asciidata = readBinaryFile("./resources/sprites/ascii_32.png");
        uint32_t width;
        uint32_t height;
        lodepng::decode(img_ascii, width, height, (const unsigned char *)asciidata->data_char,
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

    initCudaProgram(renderer, img_ascii, img_spritesheet);

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
            ImGui::RadioButton("Chunk Network", &renderMode, RENDERMODE_NETWORK_CHUNK);
            ImGui::RadioButton("Land value", &renderMode, RENDERMODE_LANDVALUE);
            ImGui::RadioButton("Flowfield", &renderMode, RENDERMODE_FLOWFIELD);

            ImGui::Text("Options");
            ImGui::Checkbox("Pring Timings", &printTimings);
            ImGui::Checkbox("Creative Mode", &creativeMode);
            ImGui::Checkbox("Display Flowfield", &displayFlowfield);
            ImGui::SliderFloat("Time multiplier", &timeMultiplier, 0.1f, 10.0f);

            if (ImGui::Button("Save Map")) {
                saveMap();
            }
            ImGui::SameLine();
            if (ImGui::Button("Load Map")) {
                loadMap();
            }

            ImGui::End();
        }
    };

    renderer->loop(update, render);

    return 0;
}
