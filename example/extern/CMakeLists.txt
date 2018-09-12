add_executable(extern extern.cpp)

target_link_libraries(extern PRIVATE arbor arbor-aux ext-tclap ext-json)
