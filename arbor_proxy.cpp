#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <numeric>

struct ArborSpike {
  std::int64_t gid;
  float time;
};
typedef std::vector< ArborSpike > ArborSpikes;

void work(int global_rank, int global_size,
	  float min_delay, float run_time)
{
  const int sbuf_length = 0;
  std::vector<int> rbuf_lengths(global_size);
  ArborSpikes rbuf;
  const std::vector<int> sbuf(1);

  for (; run_time > 0; run_time -= min_delay) {
    if (global_rank == 0) {
      std::cout << "Time left: " << run_time << std::endl;
    }
    
    MPI_Allgather(&sbuf_length, 1, MPI_INT,
		  &rbuf_lengths[0], 1, MPI_INT,
		  MPI_COMM_WORLD);

    auto total_chars = std::accumulate(rbuf_lengths.begin(),
				       rbuf_lengths.end(),
				       0);
    std::vector< int > rbuf_offset;
    rbuf_offset.reserve(global_size);
    for (int i = 0, c_offset = 0; i < global_size; i++) {
      const auto c_size = rbuf_lengths[i];
      rbuf_offset.push_back(c_offset);
      c_offset += c_size;
    }

    // spikes
    rbuf.resize(total_chars/sizeof(ArborSpike));
    MPI_Allgatherv(
		   &sbuf[0], sbuf_length, MPI_CHAR,
		   &rbuf[0], &rbuf_lengths[0], &rbuf_offset[0], MPI_CHAR,
		   MPI_COMM_WORLD);

    if (global_rank == 0) {
      for (auto spike: rbuf) {
	auto gid = spike.gid;
	auto time = spike.time;
	
	std::cout << "Gid: " << gid << ", Time: " << time << std::endl;
      }
    }
  }
}

// argv[1] = min delay
// argv[2] = run time
int main(int argc, char **argv) 
{
  float min_delay = std::stof(argv[1]);
  float run_time = std::stof(argv[2]);
  
  int rank, size;
  MPI_Init(&argc, &argv); 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  /* Build intra-communicator for local sub-group */
  MPI_Comm   intracomm;  /* intra-communicator of local sub-group */ 
  MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &intracomm);

  int lrank, lsize;
  MPI_Comm_rank(intracomm, &lrank);
  MPI_Comm_size(intracomm, &lsize);
  
  // work
  work(rank, size, min_delay, run_time);

  // cleanup
  MPI_Comm_free(&intracomm);  
  MPI_Finalize();
  
  return 0;
} 

