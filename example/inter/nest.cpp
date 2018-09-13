/*
 * A miniapp that demonstrates using an external spike source.
 */

#include <fstream>
#include <iomanip>
#include <iostream>

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

#include "parameters.hpp"

#ifdef ARB_MPI_ENABLED
#include <mpi.h>
#include <aux/with_mpi.hpp>
#endif

using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_member_type;
using arb::cell_kind;
using arb::time_type;
using arb::cell_probe_address;

int main(int argc, char** argv) {
    try {
        aux::with_mpi guard(argc, argv, false);

        //
        //  INITIALISE MPI
        //

        // split MPI_COMM_WORLD: all nest go into split 0
        MPI_Comm nest_comm;
        MPI_Comm_split(MPI_COMM_WORLD, 1, 0, &nest_comm);

        int global_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

        int arb_root = 0;

        int nest_rank;
        int nest_size;
        MPI_Comm_rank(nest_comm, &nest_rank);
        MPI_Comm_size(nest_comm, &nest_size);

        std::cout << "NEST: " << nest_rank << " of " << nest_size << std::endl;

        //
        //  HAND SHAKE ARBOR-NEST
        //

        // Get simulation length from Arbor
        float sim_duration;
        float min_delay;
        {
            int tag = 42;
            MPI_Status status;
            MPI_Recv(&sim_duration, 1, MPI_FLOAT, arb_root, tag, MPI_COMM_WORLD, &status);
            MPI_Recv(&min_delay,    1, MPI_FLOAT, arb_root, tag, MPI_COMM_WORLD, &status);
        }
        std::cout << "NEST: tfinal min_delay " << sim_duration << " " << min_delay << std::endl;

        float delta = min_delay/2;
        unsigned steps = sim_duration/delta;
        if (steps*delta<sim_duration) ++steps;

        //
        //  SEND SPIKES TO ARBOR (RUN SIMULATION)
        //

        for (unsigned step=0; step<=steps; ++step) {
            std::cout << "NEST: callback " << step << " at t " << step*delta << std::endl;

            // STEP 1: tell everyone how many spikes
            int nspikes = step==0? 1: 0;
            if (nspikes) std::cout << "NEST: sending " << nspikes << std::endl;
            MPI_Bcast(&nspikes, 1, MPI_INT, global_rank, MPI_COMM_WORLD);
            if (!nspikes) continue;

            // STEP 2: allocate memory for spikes
            std::vector<arb::spike> spikes(nspikes);
            spikes[0].source = {10u, 0u};
            spikes[0].time = 0.f;

            // STEP 3: gather spikes
            MPI_Bcast(spikes.data(), nspikes*sizeof(arb::spike), MPI_CHAR, global_rank, MPI_COMM_WORLD);
            std::cout << "NEST: callback finished " << step*delta << std::endl;
        }
    }
    catch (std::exception& e) {
        std::cerr << "exception caught in ring miniapp:\n" << e.what() << "\n";
        return 1;
    }

    return 0;
}
