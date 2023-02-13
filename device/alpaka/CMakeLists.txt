# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-cpp )

if(alpaka_ACC_GPU_CUDA_ENABLE)
  include( traccc-compiler-options-cuda )
  enable_language(CUDA)
endif()

traccc_add_library( traccc_alpaka alpaka TYPE SHARED
  # Utility definitions.
  "include/traccc/alpaka/utils/definitions.hpp"
  "include/traccc/alpaka/utils/make_prefix_sum_buff.hpp"
  "src/utils/make_prefix_sum_buff.cu"
  # Seed finding code.
  "include/traccc/alpaka/seeding/spacepoint_binning.hpp"
  "src/seeding/spacepoint_binning.cu"
)

target_link_libraries( traccc_alpaka
  PUBLIC traccc::core vecmem::core
  PRIVATE traccc::device_common alpaka::alpaka)
