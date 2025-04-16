/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/alpaka/clusterization/measurement_sorting_algorithm.hpp"

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

// oneDPL include(s).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#pragma clang diagnostic ignored "-Wimplicit-int-float-conversion"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wold-style-cast"
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#pragma clang diagnostic pop

// SYCL include(s).
#include <sycl/sycl.hpp>

namespace traccc::alpaka {

measurement_sorting_algorithm::measurement_sorting_algorithm(
    const traccc::memory_resource& mr, vecmem::copy& copy, queue& q,
    std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)), m_mr{mr}, m_copy{copy}, m_queue{q} {}

measurement_sorting_algorithm::output_type
measurement_sorting_algorithm::operator()(
    const measurement_collection_types::view& measurements_view) const {

    // TODO: This will be needed for setting up thrust / oneDPL later.
    // // Get a convenience variable for the queue that we'll be using.
    // auto queue = details::get_queue(m_queue);

    // Get the number of measurements. This is necessary because the input
    // container may not be fixed sized. And we can't give invalid pointers /
    // iterators to Thrust.
    const measurement_collection_types::view::size_type n_measurements =
        m_copy.get().get_size(measurements_view);

    // Sort the measurements in place
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    auto stream = reinterpret_cast<cudaStream_t>(
        m_queue.get().deviceNativeQueue());
    auto execPolicy =
        thrust::cuda::par_nosync(std::pmr::polymorphic_allocator(&(m_mr.main)))
            .on(stream);
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    auto stream = reinterpret_cast<hipStream_t>(
        m_queue.get().deviceNativeQueue());
    auto execPolicy = thrust::hip_rocprim::par_nosync(
        std::pmr::polymorphic_allocator(&(m_mr.main)))
        .on(stream);
#elif defined(ALPAKA_ACC_SYCL_ENABLED)
    auto syclQueue = *(reinterpret_cast<const sycl::queue*>(
        m_queue.get().deviceNativeQueue()));
    auto execPolicy = oneapi::dpl::execution::device_policy{syclQueue};
#else
    auto execPolicy = thrust::host;
#endif

#if defined(ALPAKA_ACC_SYCL_ENABLED)
    oneapi::dpl::sort(execPolicy, measurements_view.ptr(),
                      measurements_view.ptr() + n_measurements,
                      measurement_sort_comp());
#else
    thrust::sort(execPolicy, measurements_view.ptr(),
                 measurements_view.ptr() + n_measurements,
                 measurement_sort_comp());
#endif

    // Return the view of the sorted measurements.
    return measurements_view;
}

}  // namespace traccc::alpaka
