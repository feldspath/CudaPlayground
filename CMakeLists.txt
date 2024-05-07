# cmake 3.26 is required on Linux to find CUDA::nvJitLlink library 
cmake_minimum_required(VERSION 3.26)

project(CudaPlayground)
set(CMAKE_CXX_STANDARD 20)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


# add_subdirectory(./libs/openvr-master)


include(cmake/common.cmake)

##############################################
## SUBPROJECT: Random Numbers
##############################################

set(TARGET_NAME Random_Numbers) 

add_executable(${TARGET_NAME} 
	./modules/randomNumbers/main_randomNumbers.cpp
	./include/unsuck_platform_specific.cpp
)

if (MSVC)
# set workdir to ../$(SolutionDir), so that .cu files can be monitored/reloaded/compiled
set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")
endif()

target_include_directories(${TARGET_NAME} PRIVATE ./include)

# add some third-party libraries
ADD_CUDA(${TARGET_NAME})


##############################################
## SUBPROJECT: Sort Numbers
##############################################

set(TARGET_NAME Sort_Numbers) 

add_executable(${TARGET_NAME} 
	./modules/sort/main_sort.cpp
	./include/unsuck_platform_specific.cpp
)

if (MSVC)
# set workdir to ../$(SolutionDir), so that .cu files can be monitored/reloaded/compiled
set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")
endif()

target_include_directories(${TARGET_NAME} PRIVATE ./include)

# add some third-party libraries
ADD_CUDA(${TARGET_NAME})


##############################################
## SUBPROJECT: Basic Triangle Rasterizer
##############################################

set(TARGET_NAME Basic_Triangle_Rasterizer) 

add_executable( ${TARGET_NAME}
	./modules/rasterizeTriangles/main_rasterize_triangles.cpp
	./src/GLRenderer.cpp
	include/unsuck_platform_specific.cpp
)

if (MSVC)
# set workdir to ../$(SolutionDir), so that .cu files can be monitored/reloaded/compiled
set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")
endif()

target_include_directories(${TARGET_NAME} PRIVATE ./include)

# add some third-party libraries
ADD_CUDA(${TARGET_NAME})
ADD_OPENGL(${TARGET_NAME})
ADD_GLM(${TARGET_NAME})
ADD_IMGUI(${TARGET_NAME})
ADD_IMPLOT(${TARGET_NAME})


##############################################
## SUBPROJECT: Seascape
##############################################

set(TARGET_NAME Seascape) 

add_executable( ${TARGET_NAME}
	./modules/seascape/main_seascape.cpp
	./src/GLRenderer.cpp
	include/unsuck_platform_specific.cpp
)

# set workdir to ../$(SolutionDir), so that .cu files can be monitored/reloaded/compiled
if (MSVC)
set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")
endif()

target_include_directories(${TARGET_NAME} PRIVATE ./include)

# add some third-party libraries
ADD_CUDA(${TARGET_NAME})
ADD_OPENGL(${TARGET_NAME})
ADD_GLM(${TARGET_NAME})
ADD_IMGUI(${TARGET_NAME})
ADD_IMPLOT(${TARGET_NAME})

##############################################
## SUBPROJECT: VR Voxelpainter
##############################################

# Only build VR_Voxelpainter for Visual Studio
if(MSVC)
set(TARGET_NAME VR_Voxelpainter) 

add_executable( ${TARGET_NAME}
	./modules/VR_voxelpainter/main_vr.cpp
	./modules/VR_Voxelpainter/OpenVRHelper.cpp
	./src/GLRenderer.cpp
	./include/unsuck_platform_specific.cpp
)

# set workdir to ../$(SolutionDir), so that .cu files can be monitored/reloaded/compiled
set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")

target_include_directories(${TARGET_NAME} PRIVATE ./include)

# add some third-party libraries
ADD_CUDA(${TARGET_NAME})
ADD_OPENGL(${TARGET_NAME})
ADD_GLM(${TARGET_NAME})
ADD_IMGUI(${TARGET_NAME})
ADD_IMPLOT(${TARGET_NAME})
target_include_directories(${TARGET_NAME} PRIVATE ./include)

# I give up, let's just include OpenVR like this.
target_include_directories(${TARGET_NAME} PRIVATE ./libs/openvr/headers)
target_link_libraries(${TARGET_NAME} ../libs/openvr/lib/win64/openvr_api)
add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy "$(SolutionDir)../libs/openvr/openvr_api.dll" "$(OutDir)"
	COMMENT "POST BUILD: copying modules folder")
endif()


##############################################
## SOME FINAL STUFF
##############################################

if(MSVC)
# set startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Random_Numbers)
endif()