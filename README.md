# nest-arbor-proxy

Uses:

git@github.com:apeyser/nest-simulator.git

feature/nestio/arbor

cmake $path -DCMAKE_INSTALL_PREFIX=$other_path -Dwith-mpi=ON
  
Find CPATH from python -c 'import sys; print("\n".join(sys.path))'

export CPATH=/usr/lib/python3.7/site-packages

export CFLAGS="-g" 

make -j4 && make install
