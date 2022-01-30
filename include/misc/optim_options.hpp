/*################################################################################
  ##
  ##   Copyright (C) 2016-2022 Keith O'Hara
  ##
  ##   This file is part of the OptimLib C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/

#pragma once

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

// version

#ifndef OPTIM_VERSION_MAJOR
    #define OPTIM_VERSION_MAJOR 3
#endif

#ifndef OPTIM_VERSION_MINOR
    #define OPTIM_VERSION_MINOR 0
#endif

#ifndef OPTIM_VERSION_PATCH
    #define OPTIM_VERSION_PATCH 0
#endif

//

#ifdef _MSC_VER
    #error OptimLib: MSVC is not supported
#endif

//

#if defined(_OPENMP) && !defined(OPTIM_DONT_USE_OPENMP)
    #undef OPTIM_USE_OPENMP
    #define OPTIM_USE_OPENMP
#endif

#if !defined(_OPENMP) && defined(OPTIM_USE_OPENMP)
    #undef OPTIM_USE_OPENMP

    #undef OPTIM_DONE_USE_OPENMP
    #define OPTIM_DONE_USE_OPENMP
#endif

// #ifdef OPTIM_USE_OPENMP
    // #include "omp.h" //  OpenMP
// #endif

#ifdef OPTIM_DONT_USE_OPENMP
    #ifdef OPTIM_USE_OPENMP
        #undef OPTIM_USE_OPENMP
    #endif
#endif

//

#ifndef optimlib_inline
    #define optimlib_inline 
#endif

// floating point number type

#ifndef OPTIM_FPN_TYPE
    #define OPTIM_FPN_TYPE double
#endif

#if OPTIM_FPN_TYPE == float
    #undef OPTIM_FPN_SMALL_NUMBER
    #define OPTIM_FPN_SMALL_NUMBER fp_t(1e-05)
#elif OPTIM_FPN_TYPE == double
    #undef OPTIM_FPN_SMALL_NUMBER
    #define OPTIM_FPN_SMALL_NUMBER fp_t(1e-08)
#else
    #error OptimLib: floating-point number type (OPTIM_FPN_TYPE) must be 'float' or 'double'
#endif

//

namespace optim
{
    using uint_t = unsigned int;
    using fp_t = OPTIM_FPN_TYPE;

    using rand_engine_t = std::mt19937_64;

    static const double eps_dbl = std::numeric_limits<fp_t>::epsilon();
    static const double inf = std::numeric_limits<fp_t>::infinity();
}

//

#if defined(OPTIM_ENABLE_ARMA_WRAPPERS) || defined(OPTIM_USE_RCPP_ARMADILLO)
    #ifndef OPTIM_ENABLE_ARMA_WRAPPERS
        #define OPTIM_ENABLE_ARMA_WRAPPERS
    #endif

    #ifdef OPTIM_USE_RCPP_ARMADILLO
        #include <RcppArmadillo.h>
    #else
        #ifndef ARMA_DONT_USE_WRAPPER
            #define ARMA_DONT_USE_WRAPPER
        #endif
        
        #include "armadillo"
    #endif

    #ifdef OPTIM_USE_OPENMP
        #ifndef ARMA_USE_OPENMP
            #define ARMA_USE_OPENMP
        #endif
    #endif

    #ifdef OPTIM_DONT_USE_OPENMP
        #ifndef ARMA_DONT_USE_OPENMP
            #define ARMA_DONT_USE_OPENMP
        #endif
    #endif

    #ifndef BMO_ENABLE_ARMA_WRAPPERS
        #define BMO_ENABLE_ARMA_WRAPPERS
    #endif

    namespace optim
    {
        using Mat_t = arma::Mat<fp_t>;
        using ColVec_t = arma::Col<fp_t>;
        using RowVec_t = arma::Row<fp_t>;
        using ColVecInt_t = arma::Col<int>;
        using RowVecInt_t = arma::Row<int>;
        using ColVecUInt_t = arma::Col<unsigned long long>;
    }
#elif defined(OPTIM_ENABLE_EIGEN_WRAPPERS) || defined(OPTIM_USE_RCPP_EIGEN)
    #ifndef OPTIM_ENABLE_EIGEN_WRAPPERS
        #define OPTIM_ENABLE_EIGEN_WRAPPERS
    #endif
    
    #include <iostream>

    #ifdef OPTIM_USE_RCPP_EIGEN
        #include <RcppEigen.h>
    #else
        #include <Eigen/Dense>
    #endif

    #ifndef BMO_ENABLE_EIGEN_WRAPPERS
        #define BMO_ENABLE_EIGEN_WRAPPERS
    #endif

    template<typename eT, int iTr, int iTc>
    using EigenMat = Eigen::Matrix<eT,iTr,iTc>;

    namespace optim
    {
        using Mat_t = Eigen::Matrix<fp_t, Eigen::Dynamic, Eigen::Dynamic>;
        using ColVec_t = Eigen::Matrix<fp_t, Eigen::Dynamic, 1>;
        using RowVec_t = Eigen::Matrix<fp_t, 1, Eigen::Dynamic>;
        using ColVecInt_t = Eigen::Matrix<int, Eigen::Dynamic, 1>;
        using RowVecInt_t = Eigen::Matrix<int, 1, Eigen::Dynamic>;
        using ColVecUInt_t = Eigen::Matrix<size_t, Eigen::Dynamic, 1>;
    }
#else
    #error OptimLib: you must enable the Armadillo OR Eigen wrappers
#endif

//

#ifndef BMO_ENABLE_EXTRA_FEATURES
    #define BMO_ENABLE_EXTRA_FEATURES
#endif

#ifndef BMO_ENABLE_STATS_FEATURES
    #define BMO_ENABLE_STATS_FEATURES
#endif

#ifndef BMO_RNG_ENGINE_TYPE
    #define BMO_RNG_ENGINE_TYPE optim::rand_engine_t
#endif
