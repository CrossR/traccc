/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "opaque_queue.hpp"
#include "traccc/alpaka/utils/get_vecmem_resource.hpp"

namespace traccc::alpaka::details {

opaque_queue::opaque_queue(std::size_t device) : m_device{device}, m_queue(nullptr) {
    auto devAcc = ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, device);
    m_queue = std::make_unique<Queue>(devAcc);

    // Get the native queue.
    auto queue = ::alpaka::getNativeHandle(*m_queue);

    if constexpr (traccc::alpaka::pointer_type_queue::value) {
        // The native queue is a pointer to the queue.
        m_deviceNativeQueue = reinterpret_cast<void*>(queue);
    } else {
        // The native queue is a reference to the queue.
        m_deviceNativeQueue = reinterpret_cast<void*>(&queue);
    }
}

}  // namespace traccc::cuda::details
