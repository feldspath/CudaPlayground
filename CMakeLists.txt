# cmake 3.26 is required on Linux to find CUDA::nvJitLlink library 
cmake_minimum_required(VERSION 3.26)

project(CudaPlayground)
set(CMAKE_CXX_STANDARD 20)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


include(cmake/common.cmake)

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
## SUBPROJECT: Voxel Rasterizer
##############################################

set(TARGET_NAME Voxel_Rasterizer) 

add_executable( ${TARGET_NAME}
	./modules/rasterizeVoxels/main_rasterize_voxels.cpp
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
## SUBPROJECT: Shapes 2D
##############################################

set(TARGET_NAME Shapes_2D) 

add_executable( ${TARGET_NAME}
	./modules/shapes2D/main_shapes_2D.cpp
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
## SUBPROJECT: RPG
##############################################

set(TARGET_NAME rpg) 

add_executable( ${TARGET_NAME}
	./modules/rpg/main_rpg.cpp
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
ADD_LODEPNG(${TARGET_NAME})

##############################################
## SOME FINAL STUFF
##############################################

if(MSVC)
# set startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Voxel_Rasterizer)
endif()
