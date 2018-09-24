#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include <nlohmann/json.hpp>

#include <arbor/assert_macro.hpp>
#include <arbor/common_types.hpp>
#include <arbor/context.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/mc_cell.hpp>
#include <arbor/profile/meter_manager.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>
#include <arbor/recipe.hpp>
#include <arbor/version.hpp>

#include <aux/ioutil.hpp>
#include <aux/json_meter.hpp>
#include <aux/with_mpi.hpp>
#include <mpi.h>

#include "mpiutil.hpp"
#include "parameters.hpp"

int main(int argc, char **argv)
{
    try {
        aux::with_mpi guard(argc, argv, false);
        auto info = get_comm_info(true);
        auto params = read_options(argc, argv);
        on_local_rank_zero(info, [&] {
                std::cout << "ARB: starting handshake" << std::endl;
        });

        // hand shake #1: communicate cell populations and duration
        broadcast((float)params.duration, MPI_COMM_WORLD, info.arbor_root);
        broadcast((int)params.num_cells, MPI_COMM_WORLD, info.arbor_root);
        int num_nest_cells = broadcast(0,  MPI_COMM_WORLD, info.nest_root);
        int num_arbor_cells = params.num_cells;
        int total_cells = num_nest_cells + num_arbor_cells;

        on_local_rank_zero(info, [&] {
                std::cout << "ARB: num_nest_cells: " << num_nest_cells << ", "
                          << "num_arbor_cells: " << num_arbor_cells << ", "
                          << "total_cells: " << total_cells
                          << std::endl;
        });

        // hand shake #2: min delay
        float arb_comm_time = params.min_delay/2;
        broadcast(arb_comm_time, MPI_COMM_WORLD, info.arbor_root);
        float nest_comm_time = broadcast(0.f, MPI_COMM_WORLD, info.nest_root);
        float min_delay = nest_comm_time*2;
        
        float delta = min_delay/2;
        float sim_duration = params.duration;
        unsigned steps = sim_duration/delta;

        on_local_rank_zero(info, [&] {
                std::cout << "ARB: min_delay=" << min_delay << ", "
                          << "delta=" << delta << ", "
                          << "sim_duration=" << sim_duration << ", "
                          << "steps=" << steps
                          << std::endl;
        });

        std::cout << "ARB: running simulation" << std::endl;
        for (unsigned step = 0; step <= steps; ++step) {
            on_local_rank_zero(info, [&] {
                    std::cout << "ARB: callback " << step << " at t " << step*delta << std::endl;
            });

            std::vector<arb::spike> local_spikes;
            auto v = gather_spikes(local_spikes, MPI_COMM_WORLD);
            if (v.size()) print_vec_comm("ARB-recv", v, info.comm);
        }

        on_local_rank_zero(info, [&] {
                std::cout << "ARB: reached end" << std::endl;
        });
    }
    catch (std::exception& e) {
        std::cerr << "exception caught in arbor-proxy:\n" << e.what() << "\n";
        return 1;
    }

    return 0;
}

