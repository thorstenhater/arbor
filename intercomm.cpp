#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>

typedef std::vector<int> buf_type;

void work(int grank, int gsize,
	  int lrank, int lsize,
	  int rsize,
	  MPI_Comm intercomm, int right_root)
{
  // cout buffer to not overwrite
  char buf[1024*10];
  std::cout.rdbuf()->pubsetbuf(buf, sizeof(buf));

  // communications buffers
  buf_type sbuf = {grank, lrank};
  buf_type rbuf(sbuf.size()*rsize);

  // output id
  std::cout << "Pre - "
	    << "rank: "  << grank << ", "
	    << "size: "  << gsize << ", "
	    << "lrank: " << lrank << ", "
	    << "lsize: " << lsize << ", "
	    << "rsize: " << rsize << std::endl;

  // send and receive: receive length is the length of every send (??)
  MPI_Allgather(&sbuf[0], sbuf.size(), MPI_INT,
		&rbuf[0], sbuf.size(), MPI_INT,
		intercomm);

  // output other group
  std::cout << "Post - "
	    << "rank: " << grank << ", ";
  for (size_t i = 0; i < rbuf.size(); i += 2) {
    std::cout << "prank(" << i/2 << ") "
	      << rbuf[i] << ", " << rbuf[i+1]
	      << "; ";
  }
  std::cout << std::endl;

//   buf_type bufv;
//   if (grank >= right_root) {
//     bufv = {grank, lrank};
//   }

//   MPI_Allgatherv(
//   void *sendbuf,
//   int sendcount,
//   MPI_Datatype sendtype,
//   void *recvbuf,
//   int *recvcounts,
//   int *displs,
//   MPI_Datatype recvtype,
//   MPI_Comm comm
// );

}

int main(int argc, char **argv) 
{ 
  int rank, size;
  MPI_Init(&argc, &argv); 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
 
  // group: one half left, one half right
  int right_root = size/2;
  int is_left = (rank < right_root) ? 1 : 0;
 
  /* Build intra-communicator for local sub-group */
  MPI_Comm   intracomm;  /* intra-communicator of local sub-group */ 
  MPI_Comm_split(MPI_COMM_WORLD, is_left, rank, &intracomm);
 
  /* Build inter-communicators.  Tags are hard-coded. */
  MPI_Comm   intercomm;  /* inter-communicator */ 
  if (is_left) {
    MPI_Intercomm_create(intracomm, 0,
			 MPI_COMM_WORLD, right_root, 0,
			 &intercomm);
  }
  else {
    MPI_Intercomm_create(intracomm, 0,
			 MPI_COMM_WORLD, 0, 0,
			 &intercomm);
  }

  int lrank, lsize;
  MPI_Comm_rank(intracomm, &lrank);
  MPI_Comm_size(intracomm, &lsize);
  
  int rsize;
  MPI_Comm_remote_size(intercomm, &rsize);
  
  // work
  work(rank, size, lrank, lsize, rsize, intercomm, right_root);

  // cleanup
  MPI_Comm_free(&intercomm);
  MPI_Comm_free(&intracomm);  
  MPI_Finalize();
  
  return 0;
} 

