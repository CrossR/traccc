/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "utils.hpp"

// Project include(s).
#include "traccc/alpaka/utils/get_vecmem_resource.hpp"

// Standard library include(s)
#include <memory>
#include <stdexcept>

namespace traccc::alpaka::details {

// Implementation struct for vecmem_objects
struct vecmem_objects::impl {
    // Constructor
    impl() {
        // Create the appropriate resources based on the build configuration
#if defined(TRACCC_BUILD_CUDA)
        m_host_mr = std::make_unique<vecmem::cuda::host_memory_resource>();
        m_device_mr = std::make_unique<vecmem::cuda::device_memory_resource>();
        m_managed_mr = std::make_unique<vecmem::cuda::managed_memory_resource>();
        m_copy = std::make_unique<vecmem::cuda::copy>();
#elif defined(TRACCC_BUILD_HIP)
        m_host_mr = std::make_unique<vecmem::hip::host_memory_resource>();
        m_device_mr = std::make_unique<vecmem::hip::device_memory_resource>();
        m_managed_mr = std::make_unique<vecmem::hip::managed_memory_resource>();
        m_copy = std::make_unique<vecmem::hip::copy>();
#elif defined(TRACCC_BUILD_SYCL)
        m_host_mr = std::make_unique<vecmem::sycl::host_memory_resource>();
        m_device_mr = std::make_unique<vecmem::sycl::device_memory_resource>();
        m_managed_mr = std::make_unique<vecmem::sycl::shared_memory_resource>();
        m_copy = std::make_unique<vecmem::sycl::copy>();
#else
        // Default to host-only resources
        m_host_mr = std::make_unique<vecmem::host_memory_resource>();
        // For CPU-only builds, use the host memory resource for all types
        // We still use unique_ptr for consistency across all platforms
        m_device_mr = std::make_unique<vecmem::host_memory_resource>();
        m_managed_mr = std::make_unique<vecmem::host_memory_resource>();
        m_copy = std::make_unique<vecmem::copy>();
#endif
    }

    // Destructor
    ~impl() = default;

    // Resource pointers - use unique_ptr for all resources regardless of platform
    std::unique_ptr<vecmem::memory_resource> m_host_mr;
    std::unique_ptr<vecmem::memory_resource> m_device_mr;
    std::unique_ptr<vecmem::memory_resource> m_managed_mr;

    // Copy utility
    std::unique_ptr<vecmem::copy> m_copy;
};

// Constructor and destructor
vecmem_objects::vecmem_objects() : m_impl(std::make_unique<impl>()) {}
vecmem_objects::~vecmem_objects() = default;

// Implementation of the getter methods
vecmem::memory_resource& vecmem_objects::host_mr() const {
    return *(m_impl->m_host_mr);
}

vecmem::memory_resource& vecmem_objects::device_mr() const {
    return *(m_impl->m_device_mr);
}

vecmem::memory_resource& vecmem_objects::managed_mr() const {
    return *(m_impl->m_managed_mr);
}

vecmem::copy& vecmem_objects::copy() const {
    return *(m_impl->m_copy);
}

}  // namespace traccc::alpaka::details
