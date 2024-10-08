function(ADD_IMGUI TARGET_NAME)
	target_include_directories(${TARGET_NAME} PRIVATE
		libs/imgui
		libs/imgui/backends)

	target_sources(${TARGET_NAME} PRIVATE
		libs/imgui/imgui.cpp
		libs/imgui/imgui_demo.cpp
		libs/imgui/imgui_draw.cpp
		libs/imgui/imgui_tables.cpp
		libs/imgui/imgui_widgets.cpp
		libs/imgui/backends/imgui_impl_glfw.cpp
		libs/imgui/backends/imgui_impl_opengl3.cpp)
endfunction()

function(ADD_IMPLOT TARGET_NAME)
	target_include_directories(${TARGET_NAME} PRIVATE
		libs/implot)
	target_sources(${TARGET_NAME} PRIVATE
		libs/implot/implot_items.cpp
		libs/implot/implot.cpp)
endfunction()

function(ADD_GLM TARGET_NAME)
	target_include_directories(${TARGET_NAME} PRIVATE libs/glm)
endfunction()

function(ADD_LODEPNG TARGET_NAME)
	target_include_directories(${TARGET_NAME} PRIVATE libs/lodepng)
		target_sources(${TARGET_NAME} PRIVATE
		libs/lodepng/lodepng.cpp)
endfunction()

function(ADD_CUDA TARGET_NAME)
	find_package(CUDAToolkit 12.4 REQUIRED)
	target_include_directories(${TARGET_NAME} PRIVATE CUDAToolkit_INCLUDE_DIRS)
	target_link_libraries(${TARGET_NAME}
		CUDA::cuda_driver
		CUDA::nvrtc
		CUDA::nvJitLink)
	message(STATUS ${CUDAToolkit_LIBRARY_DIR})
endfunction()

function(ADD_OPENGL TARGET_NAME)
	find_package(OpenGL REQUIRED)
	target_link_libraries(${TARGET_NAME} ${OPENGL_LIBRARY})

	target_include_directories(${TARGET_NAME} PRIVATE libs/glew/include)
	target_sources(${TARGET_NAME} PRIVATE libs/glew/glew.c)
	target_compile_definitions(${TARGET_NAME} PRIVATE GLEW_STATIC)

	include(cmake/glfw.cmake)
	target_include_directories(${TARGET_NAME} PRIVATE ${glfw_SOURCE_DIR}/include)
	target_link_libraries(${TARGET_NAME} glfw)
endfunction()
