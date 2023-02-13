
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

if(alpaka_ACC_GPU_CUDA_ENABLE)
  include( traccc-compiler-options-cuda )
  find_package( CUDAToolkit REQUIRED )
  enable_language(CUDA)
endif()

traccc_add_executable( seeding_example_alpaka "seeding_example_alpaka.cu"
   LINK_LIBRARIES vecmem::core traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::alpaka alpaka::alpaka
                  traccc::options )
