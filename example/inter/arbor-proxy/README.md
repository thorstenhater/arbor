# nest-arbor-proxy

Uses:
```
git clone git@github.com:apeyser/nest-simulator.git
git clone https://github.com/apeyser/nest-simulator.git
git checkout feature/nestio/arbor

cmake $path -DCMAKE_INSTALL_PREFIX=$other_path -Dwith-mpi=ON

export CFLAGS="-g" 

make -j4 && make install
```

Setting up CPATH for mpi4py

```
python3 -c 'import sys; print("\n".join(sys.path))'
export CPATH=$CPATH:/users/bcumming/.local/lib/python3.4/site-packages/mpi4py/include/
```
