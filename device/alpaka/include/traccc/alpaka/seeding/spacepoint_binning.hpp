/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/spacepoint.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

#include "traccc/alpaka/utils/definitions.hpp"
#include "traccc/alpaka/utils/make_prefix_sum_buff.hpp"
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/seeding/device/count_grid_capacities.hpp"
#include "traccc/seeding/device/populate_grid.hpp"

// Alpaka
#include <alpaka/alpaka.hpp>

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <memory>
#include <utility>

namespace traccc::alpaka {

/// Spacepoing binning executed on an Alpaka device
template<typename TQueue, typename TAcc>
class spacepoint_binning : public algorithm<sp_grid_buffer(
                               const spacepoint_container_types::const_view&)> {

    public:
    /// Constructor for the algorithm
    spacepoint_binning(const seedfinder_config& config,
                       const spacepoint_grid_config& grid_config,
                       const traccc::memory_resource& mr,
                       TQueue& queue, TAcc const& acc) :
            m_config(config.toInternalUnits()),
            m_axes(get_axes(grid_config.toInternalUnits(),
            *(mr.host))), m_mr(mr), m_queue(queue), m_acc(acc)
    {
        m_copy = std::make_unique<vecmem::copy>();
    }

    /// Function executing the algorithm with a view of spacepoints
    sp_grid_buffer operator()(const spacepoint_container_types::const_view&
                                  spacepoints_view) const override
    {
        // Get the spacepoint sizes from the view
        auto sp_sizes = m_copy->get_sizes(spacepoints_view.items);

        // Create prefix sum buffer
        vecmem::data::vector_buffer sp_prefix_sum_buff =
            make_prefix_sum_buff(sp_sizes, *m_copy, m_mr, m_queue, m_acc);

        // Set up the container that will be filled with the required capacities for
        // the spacepoint grid.
        const std::size_t grid_bins = m_axes.first.n_bins * m_axes.second.n_bins;
        vecmem::data::vector_buffer<unsigned int> grid_capacities_buff(grid_bins,
                                                                       m_mr.main);
        m_copy->setup(grid_capacities_buff);
        m_copy->memset(grid_capacities_buff, 0);
        vecmem::data::vector_view<unsigned int> grid_capacities_view = grid_capacities_buff;

        using WorkDiv = ::alpaka::WorkDivMembers<::alpaka::DimInt<1u>, uint32_t>;
        auto const n = sp_sizes;
        auto const deviceProperties = ::alpaka::getAccDevProps<TAcc>(m_acc);
        auto const maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];
        auto const threadsPerBlock = maxThreadsPerBlock;
        auto const blocksPerGrid = deviceProperties.m_multiProcessorCount;
        auto const elementsPerThread = 1u;
        auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};

        // Fill the grid capacity container.
        auto sp_prefix_sum_buff_view = vecmem::get_data(sp_prefix_sum_buff);

        // Kokkos::parallel_for(
        //     "count_grid_capacities", team_policy(num_blocks, num_threads),
        //     KOKKOS_LAMBDA(const member_type& team_member) {
        //         device::count_grid_capacities(
        //             team_member.league_rank() * team_member.team_size() +
        //                 team_member.team_rank(),
        //             m_config, m_axes.first, m_axes.second, spacepoints_view,
        //             sp_prefix_sum_buff_view, grid_capacities_view);
        //     });
        ::alpaka::exec<TAcc>(
                m_queue,
                workDiv,
                [] ALPAKA_FN_ACC(
                    TAcc const& acc,
                    const seedfinder_config& config,
                    const sp_grid::axis_p0_type& phi_axis,
                    const sp_grid::axis_p1_type& z_axis,
                    const spacepoint_container_types::const_view& spacepoints_view,
                    vecmem::data::vector_view<device::prefix_sum_element_t>& sp_prefix_sum_buff_view,
                    vecmem::data::vector_view<unsigned int>& grid_capacities_view
                ) -> void
                {
                    auto const globalThreadExtent(::alpaka::getWorkDiv<::alpaka::Grid, ::alpaka::Threads>(acc)[0u]);
                    auto const globalThreadIdx(::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u]);

                    for (uint32_t dataDomainIdx = globalThreadIdx;
                         dataDomainIdx < spacepoints_view.items.size();
                         dataDomainIdx += globalThreadIdx)
                    {
                        device::count_grid_capacities(dataDomainIdx, config,
                            phi_axis, z_axis, spacepoints_view,
                            sp_prefix_sum_buff_view, grid_capacities_view);
                   }
                },
                m_axes.first, m_axes.second,
                spacepoints_view, sp_prefix_sum_buff_view, grid_capacities_view
            );
        ::alpaka::wait(m_queue);

        // Copy grid capacities back to the host
        vecmem::vector<unsigned int> grid_capacities_host(m_mr.host ? m_mr.host
                                                                    : &(m_mr.main));
        (*m_copy)(grid_capacities_buff, grid_capacities_host);

        // // Create the grid buffer.
        sp_grid_buffer grid_buffer(
            m_axes.first, m_axes.second,
            std::vector<std::size_t>(grid_capacities_host.begin(),
                                     grid_capacities_host.end()),
            std::vector<std::size_t>(1, 10),
            m_mr.main, m_mr.host);
        m_copy->setup(grid_buffer._buffer);
        sp_grid_view grid_view = grid_buffer;

        // // Populate the grid.
        // Kokkos::parallel_for(
        //     "populate_grid", team_policy(num_blocks, num_threads),
        //     KOKKOS_LAMBDA(const member_type& team_member) {
        //         device::populate_grid(
        //             team_member.league_rank() * team_member.team_size() +
        //                 team_member.team_rank(),
        //             m_config, spacepoints_view, sp_prefix_sum_buff_view, grid_view);
        //     });

        ::alpaka::exec<TAcc>(
                m_queue,
                workDiv,
                [] ALPAKA_FN_ACC(
                    TAcc const& acc,
                    const seedfinder_config& config,
                    const spacepoint_container_types::const_view& spacepoints_view,
                    vecmem::data::vector_view<device::prefix_sum_element_t>& sp_prefix_sum_buff_view,
                    sp_grid_view& grid_view
                ) -> void
                {
                    auto globalThreadExtent(::alpaka::getWorkDiv<::alpaka::Grid, ::alpaka::Threads>(acc)[0u]);
                    auto globalThreadIdx(::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0u]);

                    for (uint32_t dataDomainIdx = globalThreadIdx;
                         dataDomainIdx < spacepoints_view.items.size();
                         dataDomainIdx += globalThreadIdx)
                    {
                        device::populate_grid(dataDomainIdx, config,
                            spacepoints_view, sp_prefix_sum_buff_view, grid_view);
                   }
                },
                m_config, spacepoints_view, sp_prefix_sum_buff_view, grid_view
            );
        ::alpaka::wait(m_queue);

        // Return the freshly filled buffer.
        return grid_buffer;
    }

    private:
    /// Member variables
    seedfinder_config m_config;
    std::pair<sp_grid::axis_p0_type, sp_grid::axis_p1_type> m_axes;
    traccc::memory_resource m_mr;
    std::unique_ptr<vecmem::copy> m_copy;
    TQueue m_queue;
    TAcc m_acc;

};  // class spacepoint_binning

}  // namespace traccc::alpaka
