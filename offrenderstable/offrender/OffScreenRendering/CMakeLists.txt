
cmake_minimum_required(VERSION 2.8)

PROJECT(OffScreenRendering)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})


if(VTK_LIBRARIES)
  target_link_libraries(OffScreenRendering ${VTK_LIBRARIES})
else()
  target_link_libraries(OffScreenRendering vtkHybrid)
endif()

find_package(CUDA)
if (CUDA_FOUND)
        include(FindCUDA)
        set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode
arch=compute_35,code=sm_35)
        message("CUDA found")
else()
        message("CUDA not found")
endif()

cuda_add_executable(OffScreenRendering
        Coder.cu
		OffScreenRendering.cxx
)

