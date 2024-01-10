cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(optim LANGUAGES CXX)

set(ARCHITECTURE "x86" CACHE STRING "CPU Architecture [x86, ARM]")
set(OPENMP ON CACHE BOOL "Enable OpenMP parallelization")
set(FP_TYPE "double" CACHE STRING "Default floating point type")
set(LINALG_LIB "eigen" CACHE STRING "Choice of linear algebra library [arma, eigen]")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "The build type")

include(FetchContent)

# Dependencies

if(LINALG_LIB STREQUAL "arma")
    find_package(Armadillo REQUIRED)
else()
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)
    FetchContent_MakeAvailable(Eigen)
endif()

if(OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# Optimization flags

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g)
else() # default to release
    add_compile_options(
        -O3 -ffp-contract=fast -flto -DNDEBUG -fPIC 
    )

    if(ARCHITECTURE STREQUAL "ARM")
        add_compile_options(-mcpu=native)
    else() # default to x86
        add_compile_options(-march=native)
    endif()
endif()

# Other flags

if(LINALG_LIB STREQUAL "arma")
    add_compile_options(-DOPTIM_ENABLE_ARMA_WRAPPERS -DARMA_NO_DEBUG)
else() # default to eigen
    add_compile_options(-DOPTIM_ENABLE_EIGEN_WRAPPERS)
endif()

add_compile_options(-Wall -DOPTIM_FPN_TYPE=${FP_TYPE} -lblas -llapack)

# Set up targets

add_library(optim SHARED
    src/line_search/more_thuente.cpp
    src/unconstrained/de.cpp
    src/unconstrained/nm.cpp
    src/unconstrained/newton.cpp
    src/unconstrained/cg.cpp
    src/unconstrained/bfgs.cpp
    src/unconstrained/de_prmm.cpp
    src/unconstrained/pso.cpp
    src/unconstrained/pso_dv.cpp
    src/unconstrained/gd.cpp
    src/unconstrained/lbfgs.cpp
    src/zeros/broyden.cpp
    src/zeros/broyden_df.cpp
    src/constrained/sumt.cpp
)
target_include_directories(optim PUBLIC include)
if(LINALG_LIB STREQUAL "arma")
    target_include_directories(optim PRIVATE ${ARMADILLO_INCLUDE_DIRS})
    target_link_libraries(optim PRIVATE ${ARMADILLO_LIBRARIES})
else()
    target_link_libraries(optim PRIVATE Eigen3::Eigen)
endif()
if(OPENMP)
    target_link_libraries(optim PRIVATE OpenMP::OpenMP_CXX)
endif()
