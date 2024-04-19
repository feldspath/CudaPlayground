cmake_minimum_required(VERSION 3.22)

project(CudaPlayground)
set(CMAKE_CXX_STANDARD 20)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(cmake/common.cmake)

##############################################
## SUBPROJECT: Random Numbers
##############################################

set(TARGET_NAME Random_Numbers) 

add_executable(${TARGET_NAME} 
	src/main.cpp
	include/unsuck_platform_specific.cpp
)

# set workdir to ../$(SolutionDir), so that .cu files can be monitored/reloaded/compiled
set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")

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

# set workdir to ../$(SolutionDir), so that .cu files can be monitored/reloaded/compiled
set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")

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

# set workdir to ../$(SolutionDir), so that .cu files can be monitored/reloaded/compiled
set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")

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
set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")

target_include_directories(${TARGET_NAME} PRIVATE ./include)

# add some third-party libraries
ADD_CUDA(${TARGET_NAME})
ADD_OPENGL(${TARGET_NAME})
ADD_GLM(${TARGET_NAME})
ADD_IMGUI(${TARGET_NAME})
ADD_IMPLOT(${TARGET_NAME})

##############################################
## SOME FINAL STUFF
##############################################

# set startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Random_Numbers)