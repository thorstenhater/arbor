#! /usr/bin/python

from mpi4py import MPI

comm = MPI.COMM_WORLD.Split(1)
print(comm)

#import os
#os.environ['DELAY_PYNEST_INIT'] = "1"
import nest

nest.set_communicator(comm)


