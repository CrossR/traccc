
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

if(alpaka_ACC_GPU_CUDA_ENABLE)
  enable_language(CUDA)
  set_source_files_properties(seeding_example_alpaka.cpp PROPERTIES LANGUAGE CUDA)
endif()

traccc_add_executable( seeding_example_alpaka "seeding_example_alpaka.cpp"
   LINK_LIBRARIES vecmem::core traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::alpaka alpaka::alpaka
                  traccc::options )
