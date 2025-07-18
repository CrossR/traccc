/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/clusterization/measurement_sorting_algorithm.hpp"

#include "../utils/get_queue.hpp"

// Thrust include(s).
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

// System include(s).
#include <memory_resource>

namespace traccc::alpaka {

measurement_sorting_algorithm::measurement_sorting_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, queue& q,
    std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), m_mr{mr}, m_copy{copy}, m_queue{q} {}

measurement_sorting_algorithm::output_type
measurement_sorting_algorithm::operator()(
    const measurement_collection_types::view& measurements_view) const {

    // Get a convenience variable for the queue that we'll be using.
    auto queue = details::get_queue(m_queue);

    // Get the number of measurements. This is necessary because the input
    // container may not be fixed sized. And we can't give invalid pointers /
    // iterators to Thrust.
    const measurement_collection_types::view::size_type n_measurements =
        m_copy.get().get_size(measurements_view);

    // Sort the measurements in place
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    auto stream = ::alpaka::getNativeHandle(queue);
    auto execPolicy =
        thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(m_mr.main)))
            .on(stream);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    auto stream = ::alpaka::getNativeHandle(queue);
    auto execPolicy =
        thrust::hip_rocprim::par_nosync(
            std::pmr::polymorphic_allocator<std::byte>(&(m_mr.main)))
            .on(stream);
#else
    auto execPolicy = thrust::host;
#endif

    thrust::sort(execPolicy, measurements_view.ptr(),
                 measurements_view.ptr() + n_measurements,
                 measurement_sort_comp());

    // Return the view of the sorted measurements.
    return measurements_view;
}

}  // namespace traccc::alpaka
