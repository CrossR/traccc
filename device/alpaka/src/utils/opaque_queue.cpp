/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "opaque_queue.hpp"

namespace traccc::alpaka::details {

opaque_queue::opaque_queue(std::size_t device) : m_device{device}, m_queue(nullptr) {
    auto devAcc = ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, device);
    m_queue = std::make_unique<Queue>(devAcc);

#ifdef ALPAKA_ACC_SYCL_ENABLED
    auto syclQueue = ::alpaka::getNativeHandle(*m_queue);
    m_deviceNativeQueue = &syclQueue;
#else
    m_deviceNativeQueue = ::alpaka::getNativeHandle(*m_queue);
#endif
}

}  // namespace traccc::cuda::details
