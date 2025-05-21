/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/alpaka/fitting/fitting_algorithm.hpp"

#include "../utils/get_queue.hpp"
#include "../utils/parallel_algorithms.hpp"
#include "../utils/utils.hpp"
#include "traccc/fitting/device/fill_sort_keys.hpp"
#include "traccc/fitting/device/fit.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/utils/bfield.hpp"
#include "traccc/utils/detector_type_utils.hpp"

// detray include(s).
#include <detray/propagator/rk_stepper.hpp>

// System include(s).
#include <memory_resource>
#include <vector>

namespace traccc::alpaka {

struct FillSortKeysKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        track_candidate_container_types::const_view track_candidates_view,
        vecmem::data::vector_view<device::sort_key> keys_view,
        vecmem::data::vector_view<unsigned int> ids_view) const {

        device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::fill_sort_keys(globalThreadIdx, track_candidates_view,
                               keys_view, ids_view);
    }
};

template <typename fitter_t>
struct FitTrackKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const typename fitter_t::config_type cfg,
        const device::fit_payload<fitter_t>* payload) const {

        device::global_index_t globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::fit<fitter_t>(globalThreadIdx, cfg, *payload);
    }
};

template <typename fitter_t>
fitting_algorithm<fitter_t>::fitting_algorithm(
    const config_type& cfg, const traccc::memory_resource& mr,
    vecmem::copy& copy, queue& q, std::unique_ptr<const Logger> logger)
    : messaging(std::move(logger)),
      m_cfg(cfg),
      m_mr(mr),
      m_copy(copy),
      m_queue(q) {}

template <typename fitter_t>
track_state_container_types::buffer fitting_algorithm<fitter_t>::operator()(
    const typename fitter_t::detector_type::view_type& det_view,
    const typename fitter_t::bfield_type& field_view,
    const typename track_candidate_container_types::const_view&
        track_candidates_view) const {

    // Setup alpaka
    auto devHost = ::alpaka::getDevByIdx(::alpaka::Platform<Host>{}, 0u);
    auto devAcc = ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, 0u);
    auto queue = details::get_queue(m_queue);
    Idx threadsPerBlock = getWarpSize<Acc>() * 2;

    // Number of tracks
    const track_candidate_container_types::const_device::header_vector::
        size_type n_tracks = m_copy.get_size(track_candidates_view.headers);

    // Get the sizes of the track candidates in each track
    using jagged_buffer_size_type = track_candidate_container_types::
        const_device::item_vector::value_type::size_type;
    const std::vector<jagged_buffer_size_type> candidate_sizes =
        m_copy.get_sizes(track_candidates_view.items);

    track_state_container_types::buffer track_states_buffer{
        {n_tracks, m_mr.main},
        {candidate_sizes, m_mr.main, m_mr.host,
         vecmem::data::buffer_type::resizable}};
    track_state_container_types::view track_states_view(track_states_buffer);

    std::vector<jagged_buffer_size_type> seqs_sizes(candidate_sizes.size());
    std::transform(candidate_sizes.begin(), candidate_sizes.end(),
                   seqs_sizes.begin(),
                   [this](const jagged_buffer_size_type sz) {
                       return std::max(sz * m_cfg.barcode_sequence_size_factor,
                                       m_cfg.min_barcode_sequence_capacity);
                   });
    vecmem::data::jagged_vector_buffer<detray::geometry::barcode> seqs_buffer{
        seqs_sizes, m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable};

    m_copy.setup(track_states_buffer.headers)->ignore();
    m_copy.setup(track_states_buffer.items)->ignore();
    m_copy.setup(seqs_buffer)->ignore();

    // Calculate the number of threads and thread blocks to run the track
    // fitting
    std::cout << "There is " << n_tracks << "tracks in the fitting algorithm..." << std::endl;
    if (n_tracks > 0) {
        const Idx blocksPerGrid =
            (n_tracks + threadsPerBlock - 1) / threadsPerBlock;
        auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

        vecmem::data::vector_buffer<device::sort_key> keys_buffer(n_tracks,
                                                                  m_mr.main);
        vecmem::data::vector_buffer<unsigned int> param_ids_buffer(n_tracks,
                                                                   m_mr.main);

        // Get key and value for sorting
        ::alpaka::exec<Acc>(
            queue, workDiv, FillSortKeysKernel{}, track_candidates_view,
            vecmem::get_data(keys_buffer), vecmem::get_data(param_ids_buffer));

        // Sort the key to get the sorted parameter ids
        vecmem::device_vector<device::sort_key> keys_device(keys_buffer);
        vecmem::device_vector<unsigned int> param_ids_device(param_ids_buffer);

        details::sort_by_key(m_queue, m_mr, keys_device.begin(),
                             keys_device.end(), param_ids_device.begin());

        // Prepare the payload for the track fitting
        device::fit_payload<fitter_t> payload{
            .det_data = det_view,
            .field_data = field_view,
            .track_candidates_view = track_candidates_view,
            .param_ids_view = vecmem::get_data(param_ids_buffer),
            .track_states_view = track_states_view,
            .barcodes_view = vecmem::get_data(seqs_buffer)};
        auto bufHost_fitPayload =
            ::alpaka::allocBuf<device::fit_payload<fitter_t>, Idx>(devHost, 1u);
        device::fit_payload<fitter_t>* fitPayload =
            ::alpaka::getPtrNative(bufHost_fitPayload);
        *fitPayload = payload;

        auto bufAcc_fitPayload =
            ::alpaka::allocBuf<device::fit_payload<fitter_t>, Idx>(devAcc, 1u);
        ::alpaka::memcpy(queue, bufAcc_fitPayload, bufHost_fitPayload);

        // Run the track fitting
        ::alpaka::exec<Acc>(queue, workDiv, FitTrackKernel<fitter_t>{}, m_cfg,
                            ::alpaka::getPtrNative(bufAcc_fitPayload));
    }

    ::alpaka::wait(queue);

    return track_states_buffer;
}

// Explicit template instantiation
template class fitting_algorithm<
    fitter_for_t<traccc::default_detector::device>>;
}  // namespace traccc::alpaka
