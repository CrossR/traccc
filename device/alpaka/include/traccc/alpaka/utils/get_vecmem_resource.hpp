/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>
#include <vecmem/utils/cuda/copy.hpp>

#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#include <vecmem/memory/hip/device_memory_resource.hpp>
#include <vecmem/memory/hip/host_memory_resource.hpp>
#include <vecmem/memory/hip/managed_memory_resource.hpp>
#include <vecmem/utils/hip/async_copy.hpp>
#include <vecmem/utils/hip/copy.hpp>

#elif defined(ALPAKA_ACC_SYCL_ENABLED)
#include <vecmem/memory/sycl/device_memory_resource.hpp>
#include <vecmem/memory/sycl/host_memory_resource.hpp>
#include <vecmem/memory/sycl/shared_memory_resource.hpp>
#include <vecmem/utils/sycl/async_copy.hpp>
#include <vecmem/utils/sycl/copy.hpp>

#else
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>
#endif

#include "traccc/alpaka/utils/device_tag.hpp"

// Forward declarations so we can compile the types below
namespace vecmem {
class host_memory_resource;
class copy;
namespace cuda {
class host_memory_resource;
class device_memory_resource;
class managed_memory_resource;
class copy;
class async_copy;
}  // namespace cuda
namespace hip {
class host_memory_resource;
class device_memory_resource;
class managed_memory_resource;
class copy;
class async_copy;
}  // namespace hip
namespace sycl {
class host_memory_resource;
class device_memory_resource;
class shared_memory_resource;
class copy;
class async_copy;
}  // namespace sycl
}  // namespace vecmem

namespace traccc::alpaka {

// VecMem resource types for different device tags
template <typename T>
struct host_device_types {
    using device_memory_resource = vecmem::host_memory_resource;
    using host_memory_resource = vecmem::host_memory_resource;
    using managed_memory_resource = vecmem::host_memory_resource;
    using device_copy = vecmem::copy;
    using async_copy = vecmem::copy;
    using pointer_type_queue = std::false_type;
};
template <>
struct host_device_types<::alpaka::TagGpuCudaRt> {
    using device_memory_resource = vecmem::cuda::device_memory_resource;
    using host_memory_resource = vecmem::cuda::host_memory_resource;
    using managed_memory_resource = vecmem::cuda::managed_memory_resource;
    using device_copy = vecmem::cuda::copy;
    using async_device_copy = vecmem::cuda::async_copy;
    using pointer_type_queue = std::true_type;
};
template <>
struct host_device_types<::alpaka::TagGpuHipRt> {
    using device_memory_resource = vecmem::hip::device_memory_resource;
    using host_memory_resource = vecmem::hip::host_memory_resource;
    using managed_memory_resource = vecmem::hip::managed_memory_resource;
    using device_copy = vecmem::hip::copy;
    using async_device_copy = vecmem::hip::async_copy;
    using pointer_type_queue = std::true_type;
};
template <>
struct host_device_types<::alpaka::TagCpuSycl> {
    using device_memory_resource = vecmem::sycl::device_memory_resource;
    using host_memory_resource = vecmem::sycl::host_memory_resource;
    using managed_memory_resource = vecmem::sycl::shared_memory_resource;
    using device_copy = vecmem::sycl::copy;
    using async_device_copy = vecmem::sycl::async_copy;
    using pointer_type_queue = std::false_type;
};
template <>
struct host_device_types<::alpaka::TagFpgaSyclIntel> {
    using device_memory_resource = vecmem::sycl::device_memory_resource;
    using host_memory_resource = vecmem::sycl::host_memory_resource;
    using managed_memory_resource = vecmem::sycl::shared_memory_resource;
    using device_copy = vecmem::sycl::copy;
    using async_device_copy = vecmem::sycl::async_copy;
    using pointer_type_queue = std::false_type;
};
template <>
struct host_device_types<::alpaka::TagGpuSyclIntel> {
    using device_memory_resource = vecmem::sycl::device_memory_resource;
    using host_memory_resource = vecmem::sycl::host_memory_resource;
    using managed_memory_resource = vecmem::sycl::shared_memory_resource;
    using device_copy = vecmem::sycl::copy;
    using async_device_copy = vecmem::sycl::async_copy;
    using pointer_type_queue = std::false_type;
};

using device_memory_resource =
    typename host_device_types<AccTag>::device_memory_resource;
using host_memory_resource =
    typename host_device_types<AccTag>::host_memory_resource;
using managed_memory_resource =
    typename host_device_types<AccTag>::managed_memory_resource;
using device_copy = typename host_device_types<AccTag>::device_copy;
using async_device_copy =
    typename host_device_types<AccTag>::async_device_copy;
using pointer_type_queue =
    typename host_device_types<AccTag>::pointer_type_queue;

}  // namespace traccc::alpaka
