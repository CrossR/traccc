/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/finding_config.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Event Data) Payload for the @c traccc::device::propagate_to_next_surface
/// function
template <typename propagator_t, typename bfield_t>
struct propagate_to_next_surface_payload {
    /**
     * @brief View object to the tracking detector description
     */
    typename propagator_t::detector_type::view_type det_data;

    /**
     * @brief View object to the magnetic field
     */
    bfield_t field_data;

    /**
     * @brief View object to the vector of track parameters
     */
    bound_track_parameters_collection_types::view params_view;

    /**
     * @brief View object to the vector of track parameter liveness values
     */
    vecmem::data::vector_view<unsigned int> params_liveness_view;

    /**
     * @brief View object to the access order of parameters so they are sorted
     */
    vecmem::data::vector_view<const unsigned int> param_ids_view;

    /**
     * @brief View object to the vector of candidate links
     */
    vecmem::data::vector_view<const candidate_link> links_view;

    /**
     * @brief Index in the link vector at which the current step starts
     */
    const unsigned int prev_links_idx;

    /**
     * @brief Current CKF step number
     */
    unsigned int step;

    /**
     * @brief Total number of input track parameters
     */
    unsigned int n_in_params;

    /**
     * @brief View object to the vector of tips
     */
    vecmem::data::vector_view<unsigned int> tips_view;
};

/// Function for propagating the kalman-updated tracks to the next surface
///
/// If a track finds a surface that contains measurements, its bound track
/// parameter on the surface will be used for the next step. Otherwise, the link
/// is added into the tip link container so that we can know which links in the
/// link container are the final measurements of full tracks
///
/// @param[in] globalIndex        The index of the current thread
/// @param[in] cfg                Track finding config object
/// @param[inout] payload      The function call payload
///
template <typename propagator_t, typename bfield_t>
TRACCC_HOST_DEVICE inline void propagate_to_next_surface(
    global_index_t globalIndex, const finding_config& cfg,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>& payload);


template <typename propagator_t, typename bfield_t>
void debug_propagate_to_next_surface(
    std::string title, vecmem::copy& copy,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>& payload,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>*
        previous_payload) {

    // Pull out every thing from the payload
    const auto& params_view = payload.params_view;
    const auto& params_liveness_view = payload.params_liveness_view;
    const auto& param_ids_view = payload.param_ids_view;
    const auto& links_view = payload.links_view;
    const auto& tips_view = payload.tips_view;

    const auto& prev_links_idx = payload.prev_links_idx;
    const auto& step = payload.step;
    const auto& n_in_params = payload.n_in_params;

    const bool hasPreviousPayload = previous_payload != nullptr;

    // Start the debug output
    printf("=========================\n");
    printf("========== %s - %u ============\n", title.c_str(), step);
    printf("=========================\n");

    // Print the sizes to start
    printf("Parameters:\n");
    printf("  n_in_params: %u\n", n_in_params);
    printf("  prev_links_idx: %u\n", prev_links_idx);
    printf("  step: %u\n", step);
    printf("  tips_view.size(): %u\n", copy.get_size(tips_view));
    printf("  links_view.size(): %u\n", copy.get_size(links_view));
    printf("  params_view.size(): %u\n", copy.get_size(params_view));
    printf("  params_liveness_view.size(): %u\n",
           copy.get_size(params_liveness_view));
    printf("  param_ids_view.size(): %u\n", copy.get_size(param_ids_view));

    bool links_size_changed = false;
    bool params_size_changed = false;
    bool params_liveness_size_changed = false;
    bool param_ids_size_changed = false;
    bool tips_size_changed = false;

    if (hasPreviousPayload) {
        if (copy.get_size(previous_payload->links_view) !=
            copy.get_size(links_view)) {
            links_size_changed = true;
            printf("  WARNING: links_view.size() changed from %u to %u\n",
                   copy.get_size(previous_payload->links_view),
                   copy.get_size(links_view));
        }
        if (copy.get_size(previous_payload->params_view) !=
            copy.get_size(params_view)) {
            params_size_changed = true;
            printf("  WARNING: params_view.size() changed from %u to %u\n",
                   copy.get_size(previous_payload->params_view),
                   copy.get_size(params_view));
        }
        if (copy.get_size(previous_payload->params_liveness_view) !=
            copy.get_size(params_liveness_view)) {
            params_liveness_size_changed = true;
            printf(
                "  WARNING: params_liveness_view.size() changed from %u to "
                "%u\n",
                copy.get_size(previous_payload->params_liveness_view),
                copy.get_size(params_liveness_view));
        }
        if (copy.get_size(previous_payload->param_ids_view) !=
            copy.get_size(param_ids_view)) {
            param_ids_size_changed = true;
            printf("  WARNING: param_ids_view.size() changed from %u to %u\n",
                   copy.get_size(previous_payload->param_ids_view),
                   copy.get_size(param_ids_view));
        }
        if (copy.get_size(previous_payload->tips_view) !=
            copy.get_size(tips_view)) {
            tips_size_changed = true;
            printf("  WARNING: tips_view.size() changed from %u to %u\n",
                   copy.get_size(previous_payload->tips_view),
                   copy.get_size(tips_view));
        }
    }

    // Start to actually inspect the data
    // First, make views...
    vecmem::vector<unsigned int> params_liveness_view_host;
    vecmem::vector<unsigned int> param_ids_view_host;
    vecmem::vector<unsigned int> tips_view_host;
    vecmem::vector<candidate_link> links_view_host;
    bound_track_parameters_collection_types::host params_view_host;

    // Then copy from the device to the host
    copy(params_liveness_view, params_liveness_view_host)->wait();
    copy(param_ids_view, param_ids_view_host)->wait();
    copy(tips_view, tips_view_host)->wait();
    copy(links_view, links_view_host)->wait();
    copy(params_view, params_view_host)->wait();

    // Repeat for the previous payload
    vecmem::vector<unsigned int> params_liveness_view_host_old;
    vecmem::vector<unsigned int> param_ids_view_host_old;
    vecmem::vector<unsigned int> tips_view_host_old;
    vecmem::vector<candidate_link> links_view_host_old;
    bound_track_parameters_collection_types::host params_view_host_old;

    // Copy, if we have a previous payload
    copy(hasPreviousPayload ? previous_payload->params_liveness_view
                            : params_liveness_view,
         params_liveness_view_host_old)
        ->wait();
    copy(hasPreviousPayload ? previous_payload->param_ids_view : param_ids_view,
         param_ids_view_host_old)
        ->wait();
    copy(hasPreviousPayload ? previous_payload->tips_view : tips_view,
         tips_view_host_old)
        ->wait();
    copy(hasPreviousPayload ? previous_payload->links_view : links_view,
         links_view_host_old)
        ->wait();
    copy(hasPreviousPayload ? previous_payload->params_view : params_view,
         params_view_host_old)
        ->wait();

    // Print out all the unsigned ints first...
    printf("params_liveness_view:\n");
    for (unsigned int i = 0; i < params_liveness_view_host.size(); ++i) {
        printf("  %u: %u\n", i, params_liveness_view_host[i]);

        if (hasPreviousPayload && params_liveness_size_changed == false) {
            if (params_liveness_view_host[i] !=
                params_liveness_view_host_old[i]) {
                printf(
                    "  WARNING: params_liveness_view[%u] changed from "
                    "%u to %u\n",
                    i, params_liveness_view_host_old[i],
                    params_liveness_view_host[i]);
            }
        }
    }

    printf("param_ids_view:\n");
    for (unsigned int i = 0; i < param_ids_view_host.size(); ++i) {
        printf("  %u: %u\n", i, param_ids_view_host[i]);

        if (hasPreviousPayload && param_ids_size_changed == false) {
            if (param_ids_view_host[i] != param_ids_view_host_old[i]) {
                printf("  WARNING: param_ids_view[%u] changed from %u to %u\n",
                       i, param_ids_view_host_old[i],
                       param_ids_view_host[i]);
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
                    i, links_view_host_old[i].step,
                    links_view_host[i].step);
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
    for (unsigned int i = 0; i < params_view_host.size(); ++i) {
        printf("  %u: ", i);
        auto param = params_view_host[i];
        std::cout << param << std::endl;

        if (hasPreviousPayload && params_size_changed == false) {
            if (params_view_host[i] != params_view_host_old[i]) {
                std::cout << "  WARNING: params_view[" << i << "] changed from "
                          << params_view_host_old[i] << " to "
                          << params_view_host[i] << std::endl;
            }
        }
    }

    // Find the debug output
    printf("-------------------------\n");
}

}  // namespace traccc::device

// Include the implementation.
#include "./impl/propagate_to_next_surface.ipp"
