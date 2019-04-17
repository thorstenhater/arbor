add_executable(mitral EXCLUDE_FROM_ALL mitral.cpp parameters.hpp)
add_dependencies(examples mitral)

target_link_libraries(mitral PRIVATE arbor arbor-sup ext-json)
