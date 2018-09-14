#pragma once

#include <vector>

#include <mpi.h>

#include <arbor/spike.hpp>

std::vector<arb::spike> gather_spikes(const std::vector<arb::spike>& values, MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);

    std::vector<int> counts(size);
    int n_local = values.size()*sizeof(arb::spike);
    MPI_Allgather(&n_local, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> displ(size+1);
    for (int i=0; i<size; ++i) {
        displ[i+1] = displ[i] + counts[i];
    }

    std::vector<arb::spike> buffer(displ.back()/sizeof(arb::spike));
    MPI_Allgatherv(
            const_cast<arb::spike*>(values.data()), n_local, MPI_CHAR,  // send buffer
            buffer.data(), counts.data(), displ.data(), MPI_CHAR,       // receive buffer
            comm);

    return buffer;
}

int mpi_rank(MPI_Comm c) {
    int result;
    MPI_Comm_rank(c, &result);
    return result;
}

int mpi_size(MPI_Comm c) {
    int result;
    MPI_Comm_size(c, &result);
    return result;
}

int broadcast(int local, MPI_Comm comm, int root) {
    int result = local;
    MPI_Bcast(&result, 1, MPI_INT, root, comm);
    //std::cout << "broadcast<int> " << mpi_rank(comm) << " <- " << root << " local " << local << " result " << result << std::endl;
    return result;
}

float broadcast(float local, MPI_Comm comm, int root) {
    float result = local;
    MPI_Bcast(&result, 1, MPI_FLOAT, root, comm);
    //std::cout << "broadcast<float> " << mpi_rank(comm) << " <- " << root << " local " << local << " result " << result << std::endl;
    return result;
}

struct comm_info {
    int global_size; //
    int global_rank; //
    int local_rank; //
    bool is_arbor; //
    bool is_nest;  //
    int arbor_size; //
    int nest_size; //
    int arbor_root; //
    int nest_root; //
    MPI_Comm comm; //
};

inline
std::ostream& operator<<(std::ostream& o, comm_info i) {
    return o << "global ( rank "<< i.global_rank << ", size " << i.global_size<< ")\n"
             << "local rank " << i.local_rank << "\n"
             << (i.is_arbor? "arbor": "nest") << "\n"
             << (i.is_nest? "nest": "arbor") << "\n"
             << "arbor (root " << i.arbor_root << ", size " << i.arbor_size << ")\n"
             << "nest (root " << i.nest_root << ", size " << i.nest_size << ")\n";
}

comm_info get_comm_info(bool is_arbor) {
    comm_info info;
    info.is_arbor = is_arbor;
    info.is_nest = !is_arbor;

    MPI_Comm_size(MPI_COMM_WORLD, &info.global_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &info.global_rank);

    // split MPI_COMM_WORLD: all arbor go into split 1
    int color = is_arbor? 1: 0;
    MPI_Comm_split(MPI_COMM_WORLD, color, info.global_rank, &info.comm);

    int local_size;
    MPI_Comm_size(info.comm, &local_size);
    MPI_Comm_rank(info.comm, &info.local_rank);
    info.arbor_size = is_arbor? local_size: info.global_size - local_size;

    info.nest_size = info.global_size - info.arbor_size;

    // assume that ranks [0:Nn) = nest
    // assume that ranks [Nn:N) = arbor
    info.arbor_root = info.nest_size;
    info.nest_root = 0;

    return info;
}

