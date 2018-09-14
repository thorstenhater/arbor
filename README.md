# nest-arbor-proxy

Uses:
git@github.com:apeyser/nest-simulator.git

cmake <path> -DCMAKE_INSTALL_PREFIX=<path> -Dwith-mpi=ON
  
export CPATH=/usr/lib/python3.7/site-packages
export CFLAGS="-g" 
make -j4 && make install
