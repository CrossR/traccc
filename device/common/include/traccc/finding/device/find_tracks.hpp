/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/device/concepts/thread_id.hpp"

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/finding_config.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <utility>

namespace traccc::device {

/// (Global Event Data) Payload for the @c traccc::device::find_tracks function
template <typename detector_t>
struct find_tracks_payload {
    /**
     * @brief View object to the tracking detector description
     */
    typename detector_t::view_type det_data;

    /**
     * @brief View object to the vector of bound track parameters
     *
     * @warning Measurements on the same surface must be adjacent
     */
    measurement_collection_types::const_view measurements_view;

    /**
     * @brief View object to the vector of track parameters
     */
    bound_track_parameters_collection_types::const_view in_params_view;

    /**
     * @brief View object to the vector of boolean-like integers describing the
     * liveness of each parameter
     */
    vecmem::data::vector_view<const unsigned int> in_params_liveness_view;

    /**
     * @brief The total number of input parameters
     */
    unsigned int n_in_params;

    /**
     * @brief View object to the vector of barcodes for each measurement
     */
    vecmem::data::vector_view<const detray::geometry::barcode> barcodes_view;

    /**
     * @brief View object to the vector of upper bounds of measurement indices
     * per surface
     */
    vecmem::data::vector_view<const unsigned int> upper_bounds_view;

    /**
     * @brief View object to the link vector
     */
    vecmem::data::vector_view<candidate_link> links_view;

    /**
     * @brief Index in the link vector at which the previous step starts
     */
    const unsigned int prev_links_idx;

    /**
     * @brief Index in the link vector at which the current step starts
     */
    const unsigned int curr_links_idx;

    /**
     * @brief The current step identifier
     */
    unsigned int step;

    /**
     * @brief View object to the output track parameter vector
     */
    bound_track_parameters_collection_types::view out_params_view;

    /**
     * @brief View object to the output track parameter liveness vector
     */
    vecmem::data::vector_view<unsigned int> out_params_liveness_view;

    /**
     * @brief View object to the vector of tips
     */
    vecmem::data::vector_view<unsigned int> tips_view;

    /**
     * @brief View object to the vector of the number of tracks per initial
     * input seed
     */
    vecmem::data::vector_view<unsigned int> n_tracks_per_seed_view;
};

/// (Shared Event Data) Payload for the @c traccc::device::find_tracks function
struct find_tracks_shared_payload {
    /**
     * @brief Shared-memory vector with the number of measurements found per
     * track
     */
    unsigned int* shared_num_candidates;

    /**
     * @brief Shared-memory vector of measurement candidats with ID and
     * original track parameter identifier
     *
     * @note Length is always twice the block size
     */
    std::pair<unsigned int, unsigned int>* shared_candidates;

    /**
     * @brief Shared-memory atomic variable to track the size of
     * \ref shared_candidates
     */
    unsigned int& shared_candidates_size;
};

/// Function for combinatorial finding.
/// If the chi2 of the measurement < chi2_max, its measurement index and the
/// index of the link from the previous step are added to the link container.
///
/// @param[in] thread_id          A thread identifier object
/// @param[in] barrier            A block-wide barrier
/// @param[in] cfg                Track finding config object
/// @param[inout] payload         The global memory payload
/// @param[inout] shared_payload  The shared memory payload
///
template <typename detector_t, concepts::thread_id1 thread_id_t,
          concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void find_tracks(
    const thread_id_t& thread_id, const barrier_t& barrier,
    const finding_config& cfg, const find_tracks_payload<detector_t>& payload,
    const find_tracks_shared_payload& shared_payload);

template <typename detector_t>
void debug_find_tracks(
    std::string title, vecmem::copy& copy,
    const find_tracks_payload<detector_t>& payload,
    const find_tracks_payload<detector_t>* previous_payload) {

    // Pull out every thing from the payload
    const auto& in_params_view = payload.in_params_view;
    const auto& in_params_liveness_view = payload.in_params_liveness_view;
    const auto& n_in_params = payload.n_in_params;
    const auto& barcodes_view = payload.barcodes_view;
    const auto& upper_bounds_view = payload.upper_bounds_view;
    const auto& links_view = payload.links_view;
    const auto& prev_links_idx = payload.prev_links_idx;
    const auto& curr_links_idx = payload.curr_links_idx;
    const auto& step = payload.step;
    const auto& out_params_view = payload.out_params_view;
    const auto& out_params_liveness_view = payload.out_params_liveness_view;
    const auto& tips_view = payload.tips_view;
    const auto& n_tracks_per_seed_view = payload.n_tracks_per_seed_view;

    const bool hasPreviousPayload = previous_payload != nullptr;

    // Start the debug output
    printf("=========================\n");
    printf("========== %s - %u ============\n", title.c_str(), step);
    printf("=========================\n");

    // Print the sizes to start
    printf("Parameters:\n");
    printf("  n_in_params: %u\n", n_in_params);
    printf("  prev_links_idx: %u\n", prev_links_idx);
    printf("  curr_links_idx: %u\n", curr_links_idx);
    printf("  step: %u\n", step);
    printf("  tips_view.size(): %u\n", copy.get_size(tips_view));
    printf("  links_view.size(): %u\n", copy.get_size(links_view));
    printf("  in_params_view.size(): %u\n", copy.get_size(in_params_view));
    printf("  in_params_liveness_view.size(): %u\n",
           copy.get_size(in_params_liveness_view));
    printf("  out_params_liveness_view.size(): %u\n",
           copy.get_size(out_params_liveness_view));
    printf("  out_params_view.size(): %u\n", copy.get_size(out_params_view));
    printf("  barcodes.size(): %u\n", copy.get_size(barcodes_view));
    printf("  upper_bounds.size(): %u\n", copy.get_size(upper_bounds_view));

    bool links_size_changed = false;
    bool params_size_changed = false;
    bool params_liveness_size_changed = false;
    bool out_params_size_changed = false;
    bool out_params_liveness_size_changed = false;
    bool tips_size_changed = false;
    bool n_tracks_per_seed_size_changed = false;

    if (hasPreviousPayload) {
        if (copy.get_size(previous_payload->links_view) !=
            copy.get_size(links_view)) {
            links_size_changed = true;
            printf("  WARNING: links_view.size() changed from %u to %u\n",
                   copy.get_size(previous_payload->links_view),
                   copy.get_size(links_view));
        }
        if (copy.get_size(previous_payload->in_params_view) !=
            copy.get_size(in_params_view)) {
            params_size_changed = true;
            printf("  WARNING: in_params_view.size() changed from %u to %u\n",
                   copy.get_size(previous_payload->in_params_view),
                   copy.get_size(in_params_view));
        }
        if (copy.get_size(previous_payload->in_params_liveness_view) !=
            copy.get_size(in_params_liveness_view)) {
            params_liveness_size_changed = true;
            printf(
                "  WARNING: in_params_liveness_view.size() changed from "
                "%u to %u\n",
                copy.get_size(previous_payload->in_params_liveness_view),
                copy.get_size(in_params_liveness_view));
        }
        if (copy.get_size(previous_payload->out_params_view) !=
            copy.get_size(out_params_view)) {
            out_params_size_changed = true;
            printf("  WARNING: out_params_view.size() changed from %u to %u\n",
                   copy.get_size(previous_payload->out_params_view),
                   copy.get_size(out_params_view));
        }
        if (copy.get_size(previous_payload->out_params_liveness_view) !=
            copy.get_size(out_params_liveness_view)) {
            out_params_liveness_size_changed = true;
            printf(
                "  WARNING: out_params_liveness_view.size() changed from "
                "%u to %u\n",
                copy.get_size(previous_payload->out_params_liveness_view),
                copy.get_size(out_params_liveness_view));
        }
        if (copy.get_size(previous_payload->tips_view) !=
            copy.get_size(tips_view)) {
            tips_size_changed = true;
            printf("  WARNING: tips_view.size() changed from %u to %u\n",
                   copy.get_size(previous_payload->tips_view),
                   copy.get_size(tips_view));
        }
        if (copy.get_size(previous_payload->n_tracks_per_seed_view) !=
            copy.get_size(n_tracks_per_seed_view)) {
            n_tracks_per_seed_size_changed = true;
            printf(
                "  WARNING: n_tracks_per_seed.size() changed from %u to %u\n",
                copy.get_size(previous_payload->n_tracks_per_seed_view),
                copy.get_size(n_tracks_per_seed_view));
        }
    }

    // Start to actually inspect the data
    // First, make views...
    vecmem::vector<candidate_link> links_view_host;
    bound_track_parameters_collection_types::host in_params_view_host;
    vecmem::vector<unsigned int> in_params_liveness_view_host;
    bound_track_parameters_collection_types::host out_params_view_host;
    vecmem::vector<unsigned int> out_params_liveness_view_host;
    vecmem::vector<unsigned int> tips_view_host;
    vecmem::vector<unsigned int> n_tracks_per_seed_view_host;

    // Then copy from the device to the host
    copy(links_view, links_view_host)->wait();
    copy(in_params_view, in_params_view_host)->wait();
    copy(in_params_liveness_view, in_params_liveness_view_host)->wait();
    copy(out_params_view, out_params_view_host)->wait();
    copy(out_params_liveness_view, out_params_liveness_view_host)->wait();
    copy(tips_view, tips_view_host)->wait();
    copy(n_tracks_per_seed_view, n_tracks_per_seed_view_host)->wait();

    // Repeat for the previous payload
    vecmem::vector<candidate_link> links_view_host_old;
    bound_track_parameters_collection_types::host in_params_view_host_old;
    vecmem::vector<unsigned int> in_params_liveness_view_host_old;
    bound_track_parameters_collection_types::host out_params_view_host_old;
    vecmem::vector<unsigned int> out_params_liveness_view_host_old;
    vecmem::vector<unsigned int> tips_view_host_old;
    vecmem::vector<unsigned int> n_tracks_per_seed_view_host_old;

    // Copy, if we have a previous payload
    copy(hasPreviousPayload ? previous_payload->links_view : links_view,
         links_view_host_old)
        ->wait();
    copy(hasPreviousPayload ? previous_payload->in_params_view
                            : in_params_view,
         in_params_view_host_old)
        ->wait();
    copy(hasPreviousPayload ? previous_payload->in_params_liveness_view
                            : in_params_liveness_view,
         in_params_liveness_view_host_old)
        ->wait();
    copy(hasPreviousPayload ? previous_payload->out_params_view
                            : out_params_view,
         out_params_view_host_old)
        ->wait();
    copy(hasPreviousPayload ? previous_payload->out_params_liveness_view
                            : out_params_liveness_view,
         out_params_liveness_view_host_old)
        ->wait();
    copy(hasPreviousPayload ? previous_payload->tips_view : tips_view,
         tips_view_host_old)
        ->wait();
    copy(hasPreviousPayload ? previous_payload->n_tracks_per_seed_view
                            : n_tracks_per_seed_view,
         n_tracks_per_seed_view_host_old)
        ->wait();

    // Print out all the unsigned ints first...
    printf("in_params_liveness_view:\n");
    for (unsigned int i = 0; i < in_params_liveness_view_host.size(); ++i) {
        printf("  %u: %u\n", i, in_params_liveness_view_host[i]);

        if (hasPreviousPayload && params_liveness_size_changed == false) {
            if (in_params_liveness_view_host[i] !=
                in_params_liveness_view_host_old[i]) {
                printf(
                    "  WARNING: in_params_liveness_view[%u] changed from "
                    "%u to %u\n",
                    i, in_params_liveness_view_host_old[i],
                    in_params_liveness_view_host[i]);
            }
        }
    }

    printf("out_params_liveness_view:\n");
    for (unsigned int i = 0; i < out_params_liveness_view_host.size(); ++i) {
        printf("  %u: %u\n", i, out_params_liveness_view_host[i]);

        if (hasPreviousPayload && out_params_liveness_size_changed == false) {
            if (out_params_liveness_view_host[i] !=
                out_params_liveness_view_host_old[i]) {
                printf(
                    "  WARNING: out_params_liveness_view[%u] changed from "
                    "%u to %u\n",
                    i, out_params_liveness_view_host_old[i],
                    out_params_liveness_view_host[i]);
            }
        }
    }

    printf("n_tracks_per_seed_view:\n");
    for (unsigned int i = 0; i < n_tracks_per_seed_view_host.size(); ++i) {
        printf("  %u: %u\n", i, n_tracks_per_seed_view_host[i]);

        if (hasPreviousPayload && n_tracks_per_seed_size_changed == false) {
            if (n_tracks_per_seed_view_host[i] !=
                n_tracks_per_seed_view_host_old[i]) {
                printf(
                    "  WARNING: n_tracks_per_seed_view[%u] changed from "
                    "%u to %u\n",
                    i, n_tracks_per_seed_view_host_old[i],
                    n_tracks_per_seed_view_host[i]);
            }
        }
    }

    printf("tips_view:\n");
    for (unsigned int i = 0; i < tips_view_host.size(); ++i) {
        printf("  %u: %u\n", i, tips_view_host[i]);

        if (hasPreviousPayload && tips_size_changed == false) {
            if (tips_view_host[i] != tips_view_host_old[i]) {
                printf("  WARNING: tips_view[%u] changed from %u to %u\n", i,
                       tips_view_host_old[i], tips_view_host[i]);
            }
        }
    }

    // Now the links...
    printf("links_view:\n");
    for (unsigned int i = 0; i < links_view_host.size(); ++i) {
        printf("  %u: ", i);
        auto link = links_view_host[i];
        printf(
            "step: %u, previous_candidate_idx: %u, meas_idx: %u, "
            "seed_idx: %u, n_skipped: %u, chi2: %f\n",
            link.step, link.previous_candidate_idx, link.meas_idx,
            link.seed_idx, link.n_skipped, link.chi2);

        if (hasPreviousPayload && links_size_changed == false) {
            if (links_view_host[i].step != links_view_host_old[i].step) {
                printf(
                    "  WARNING: links_view[%u].step changed from %u to "
                    "%u\n",
                    i, links_view_host_old[i].step, links_view_host[i].step);
            }
            if (links_view_host[i].previous_candidate_idx !=
                links_view_host_old[i].previous_candidate_idx) {
                printf(
                    "  WARNING: links_view[%u].previous_candidate_idx "
                    "changed from %u to %u\n",
                    i, links_view_host_old[i].previous_candidate_idx,
                    links_view_host[i].previous_candidate_idx);
            }
        }
    }

    // Finally, the parameters...
    printf("params_view:\n");
    for (unsigned int i = 0; i < in_params_view_host.size(); ++i) {
        printf("  %u: ", i);
        auto param = in_params_view_host[i];
        std::cout << param << std::endl;

        if (hasPreviousPayload && params_size_changed == false) {
            if (in_params_view_host[i] != in_params_view_host_old[i]) {
                std::cout << "  WARNING: in_params_view[" << i
                          << "] changed from " << in_params_view_host_old[i]
                          << " to " << in_params_view_host[i] << std::endl;
            }
        }
    }

    printf("out_params_view:\n");
    for (unsigned int i = 0; i < out_params_view_host.size(); ++i) {
        printf("  %u: ", i);
        auto param = out_params_view_host[i];
        std::cout << param << std::endl;

        if (hasPreviousPayload && out_params_size_changed == false) {
            if (out_params_view_host[i] != out_params_view_host_old[i]) {
                std::cout << "  WARNING: out_params_view[" << i
                          << "] changed from " << out_params_view_host_old[i]
                          << " to " << out_params_view_host[i] << std::endl;
            }
        }
    }

    // Find the debug output
    printf("-------------------------\n");
}

}  // namespace traccc::device

// Include the implementation.
#include "./impl/find_tracks.ipp"
