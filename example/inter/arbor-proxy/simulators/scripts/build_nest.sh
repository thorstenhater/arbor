nest_repo_path=$build_path/nest
nest_build_path=$nest_repo_path/build
echo ==== repo  $nest_repo_path
echo ==== build $nest_build_path

# clear log file from previous builds
out="$build_path/log_nest"
rm -rf $out

msg "NEST: starting build, see $out for log"

# only check out code if not already checked out
if [ ! -d "$nest_repo_path/.git" ]
then
    msg "NEST: cloning"
    #git clone https://github.com/nest/nest-simulator.git $nest_repo_path &>> $out
    git clone https://github.com/apeyser/nest-simulator.git $nest_repo_path &>> $out
    [ $? != 0 ] && err "see ${out}" && return 1

    cd $nest_repo_path

    msg "NEST: checkout alex's branch feature/nestio/arbor"
    git checkout feature/nestio/arbor &>> $out
    [ $? != 0 ] && err "see ${out}" && return 1
else
    msg "NEST: repository has already been checked out"
fi

cd $nest_repo_path

# comment out the documentation build in the CMake file because
# it breaks with python3 on Daint.
sed 's|^add_subdirectory( doc )|#add_subdirectory( doc )|g' "$nest_repo_path/CMakeLists.txt" -i
[ $? != 0 ] && err "unable to update CMakeLists to remove docs target" && return 1

# only configure build if not already configured
if [ ! -d "$nest_build_path" ]
then
    msg "NEST: configure build"
    mkdir -p "$nest_build_path"
    cd "$nest_build_path"
    nest_cmake_flags=-DCMAKE_INSTALL_PREFIX:PATH="$install_path"
    if [ "$with_mpi" = "true" ]; then
        nest_cmake_flags="$nest_cmake_flags -Dwith-mpi=ON";
    fi
    nest_cmake_flags="$nest_cmake_flags -Dwith-ltdl=OFF";
    cmake .. $nest_cmake_flags -DPYTHON_EXECUTABLE=/usr/bin/python3 &>> ${out}
    [ $? != 0 ] && err "see ${out}" && return 1
fi

cd "$nest_build_path"

msg "NEST: build"
make -j6 &>> ${out}
[ $? != 0 ] && err "see ${out}" && return 1

msg "NEST: install"
make install &>> ${out}
[ $? != 0 ] && err "see ${out}" && return 1

msg "NEST: build completed"

cd $base_path
