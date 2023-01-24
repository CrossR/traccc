/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/alpaka/utils/definitions.hpp"
#include <alpaka/example/ExampleDefaultAcc.hpp>

// Project include(s).
#include "traccc/device/make_prefix_sum_buffer.hpp"
#include "traccc/alpaka/utils/make_prefix_sum_buff.hpp"

namespace traccc::alpaka {

template<typename TQueue, typename TAcc>
vecmem::data::vector_buffer<device::prefix_sum_element_t> make_prefix_sum_buff(
    const std::vector<device::prefix_sum_size_t>& sizes, vecmem::copy& copy,
    const traccc::memory_resource& mr, TQueue& queue, const TAcc& acc) {

    const device::prefix_sum_buffer_t make_sum_result =
        device::make_prefix_sum_buffer(sizes, copy, mr);
    const vecmem::data::vector_view<const device::prefix_sum_size_t>
        sizes_sum_view = make_sum_result.view;
    const unsigned int totalSize = make_sum_result.totalSize;

    // Create buffer and view objects for prefix sum vector
    vecmem::data::vector_buffer<device::prefix_sum_element_t> prefix_sum_buff(
        totalSize, mr.main);
    copy.setup(prefix_sum_buff);

    using Dim = ::alpaka::Dim<TAcc>;
    using Vec = ::alpaka::Vec<Dim, size_t>;
    Vec const elementsPerThread(Vec::all(static_cast<size_t>(1)));
    Vec const threadsPerGrid(Vec::all(static_cast<size_t>(8)));
    using WorkDiv = ::alpaka::WorkDivMembers<Dim, size_t>;
    WorkDiv const workDiv = ::alpaka::getValidWorkDiv<TAcc>(
        acc,
        threadsPerGrid,
        elementsPerThread,
        false,
        ::alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);

    ::alpaka::exec<TAcc>(
            queue,
            workDiv,
            [&] ALPAKA_FN_ACC(
                TAcc const& lambdaAcc,
                vecmem::data::vector_view<const device::prefix_sum_size_t> sizes_view,
                vecmem::data::vector_view<device::prefix_sum_element_t> ps_view) -> void
            {
                auto threadIdx = ::alpaka::getIdx<::alpaka::Block, ::alpaka::Threads>(lambdaAcc)[0u];
                auto blockDim = ::alpaka::getWorkDiv<::alpaka::Block, ::alpaka::Threads>(lambdaAcc)[0u];
                auto blockIdx = ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Blocks>(lambdaAcc)[0u];

                device::fill_prefix_sum(threadIdx + blockIdx * blockDim, sizes_view, ps_view);
            },
            sizes_sum_view,
            prefix_sum_buff
        );
    ::alpaka::wait(queue);

    return prefix_sum_buff;
}

}  // namespace traccc::alpaka
