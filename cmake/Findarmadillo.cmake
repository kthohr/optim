
find_package(PkgConfig)
pkg_check_modules(PC_EIGEN eigen3)
set(EIGEN_DEFINITIONS ${PC_EIGEN_CFLAGS_OTHER})

set(ARMA_HINT_DIRS
        /usr/include
        /usr/local/include
        /opt/include
        /opt/local/include)

find_path(ARMA_INCLUDE_DIR armadillo
        PATHS ${ARMA_HINT_DIRS}
        PATH_SUFFIXES armadillo)

set(ARMA_INCLUDE_DIRS ${ARMA_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Armadello DEFAULT_MSG ARMA_INCLUDE_DIR)

mark_as_advanced(ARMA_INCLUDE_DIR)

if(ARMA_FOUND)
    message(STATUS "Armadillo found (include: ${ARMA_INCLUDE_DIRS})")
endif(ARMA_FOUND)


set(armadillo_INCLUDE_DIRS ${ARMA_INCLUDE_DIRS})
set(armadillo_FOUND ${ARMA_FOUND})
#set(Eigen_DEFINITIONS ${EIGEN_DEFINITIONS})
