add_executable(arb_inter arb.cpp)
add_executable(nest_inter nest.cpp)

target_link_libraries(arb_inter PRIVATE arbor arbor-aux ext-json)
target_link_libraries(nest_inter PRIVATE arbor arbor-aux ext-json)
