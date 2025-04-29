/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// VecMem include(s).
#if defined(TRACCC_BUILD_CUDA)
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>
#endif

#if defined(TRACCC_BUILD_HIP)
#include <vecmem/memory/hip/device_memory_resource.hpp>
#include <vecmem/memory/hip/host_memory_resource.hpp>
#include <vecmem/memory/hip/managed_memory_resource.hpp>
#include <vecmem/utils/hip/copy.hpp>
#endif

#if defined(TRACCC_BUILD_SYCL)
#include <vecmem/memory/sycl/device_memory_resource.hpp>
#include <vecmem/memory/sycl/host_memory_resource.hpp>
#include <vecmem/memory/sycl/shared_memory_resource.hpp>
#include <vecmem/utils/sycl/copy.hpp>
#endif

#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// Standard library includes
#include <memory>

// Forward declarations so we can compile the types below
namespace vecmem {
class host_memory_resource;
class copy;
namespace cuda {
class host_memory_resource;
class device_memory_resource;
class managed_memory_resource;
class copy;
}  // namespace cuda
namespace hip {
class host_memory_resource;
class device_memory_resource;
class managed_memory_resource;
class copy;
}  // namespace hip
namespace sycl {
class host_memory_resource;
class device_memory_resource;
class shared_memory_resource;
class copy;
}  // namespace sycl
}  // namespace vecmem

namespace traccc::alpaka::details {

/**
 * @brief Class that creates and owns vecmem resources, providing a generic interface
 *
 * This class uses the PIMPL pattern to hide implementation details and avoid
 * preprocessor-heavy code in client code.
 */
class vecmem_objects {
public:
    /// Constructor that initializes appropriate memory resources
    vecmem_objects();

    /// Destructor
    ~vecmem_objects();

    // Delete copy and move semantics since we manage resources
    vecmem_objects(const vecmem_objects&) = delete;
    vecmem_objects& operator=(const vecmem_objects&) = delete;
    vecmem_objects(vecmem_objects&&) = delete;
    vecmem_objects& operator=(vecmem_objects&&) = delete;

    /// Get the host memory resource
    vecmem::memory_resource& host_mr() const;

    /// Get the device memory resource
    vecmem::memory_resource& device_mr() const;

    /// Get the managed memory resource (unified memory)
    vecmem::memory_resource& managed_mr() const;

    /// Get the copy utility
    vecmem::copy& copy() const;

private:
    /// Implementation details
    struct impl;

    /// Pointer to implementation
    std::unique_ptr<impl> m_impl;
};

}  // namespace traccc::alpaka::details
