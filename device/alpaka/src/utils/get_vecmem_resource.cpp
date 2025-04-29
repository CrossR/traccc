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

// Templated struct to get the correct memory resource for the current
// accelerator type.
template <typename T>
struct host_device_types {
    using device_memory_resource = vecmem::host_memory_resource;
    using host_memory_resource = vecmem::host_memory_resource;
    using managed_memory_resource = vecmem::host_memory_resource;
    using device_copy = vecmem::copy;
};
template <>
struct host_device_types<::alpaka::TagGpuCudaRt> {
    using device_memory_resource = vecmem::cuda::device_memory_resource;
    using host_memory_resource = vecmem::cuda::host_memory_resource;
    using managed_memory_resource = vecmem::cuda::managed_memory_resource;
    using device_copy = vecmem::cuda::copy;
};
template <>
struct host_device_types<::alpaka::TagGpuHipRt> {
    using device_memory_resource = vecmem::hip::device_memory_resource;
    using host_memory_resource = vecmem::hip::host_memory_resource;
    using managed_memory_resource = vecmem::hip::managed_memory_resource;
    using device_copy = vecmem::hip::copy;
};
template <>
struct host_device_types<::alpaka::TagCpuSycl> {
    using device_memory_resource = vecmem::sycl::device_memory_resource;
    using host_memory_resource = vecmem::sycl::host_memory_resource;
    using managed_memory_resource = vecmem::sycl::shared_memory_resource;
    using device_copy = vecmem::sycl::copy;
};
template <>
struct host_device_types<::alpaka::TagFpgaSyclIntel> {
    using device_memory_resource = vecmem::sycl::device_memory_resource;
    using host_memory_resource = vecmem::sycl::host_memory_resource;
    using managed_memory_resource = vecmem::sycl::shared_memory_resource;
    using device_copy = vecmem::sycl::copy;
};
template <>
struct host_device_types<::alpaka::TagGpuSyclIntel> {
    using device_memory_resource = vecmem::sycl::device_memory_resource;
    using host_memory_resource = vecmem::sycl::host_memory_resource;
    using managed_memory_resource = vecmem::sycl::shared_memory_resource;
    using device_copy = vecmem::sycl::copy;
};

using AccTag = ::alpaka::AccToTag<Acc>;
using device_memory_resource =
    typename host_device_types<AccTag>::device_memory_resource;
using host_memory_resource =
    typename host_device_types<AccTag>::host_memory_resource;
using managed_memory_resource =
    typename host_device_types<AccTag>::managed_memory_resource;
using device_copy = typename host_device_types<AccTag>::device_copy;

// Implementation struct for vecmem_objects
struct vecmem_objects::impl {
    // Constructor
    impl() {
        m_host_mr = std::make_unique<host_memory_resource>();
        m_device_mr = std::make_unique<device_memory_resource>();
        m_managed_mr = std::make_unique<managed_memory_resource>();
        m_copy = std::make_unique<device_copy>();
    }

    // Destructor
    ~impl() = default;

    std::unique_ptr<vecmem::memory_resource> m_host_mr;
    std::unique_ptr<vecmem::memory_resource> m_device_mr;
    std::unique_ptr<vecmem::memory_resource> m_managed_mr;
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
