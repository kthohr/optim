cmake_minimum_required(VERSION 2.8.3)
project(optim)

add_definitions(-std=c++11)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(catkin REQUIRED)
find_package(armadillo REQUIRED) # /usr/include

catkin_package(
        INCLUDE_DIRS include #/unconstrained include/constrained include/line_search include/misc
        LIBRARIES ${PROJECT_NAME}
        DEPENDS armadillo
)

include_directories(include ${armadillo_INCLUDE_DIRS})

set(LIB_SRCS
        src/unconstrained/bfgs.cpp
        src/unconstrained/cg.cpp
        src/unconstrained/de.cpp
        src/unconstrained/de_prmm.cpp
        src/unconstrained/gd.cpp
        src/unconstrained/lbfgs.cpp
        src/unconstrained/newton.cpp
        src/unconstrained/nm.cpp
        src/unconstrained/pso_dv.cpp
        src/unconstrained/pso.cpp
        )
add_library(${PROJECT_NAME}
        ${LIB_SRCS}
        )


# link_directories()

install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.hpp"
        PATTERN ".svn" EXCLUDE
        )