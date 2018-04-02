/*################################################################################
  ##
  ##   Copyright (C) 2016-2018 Keith O'Hara
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

// version

#ifndef OPTIM_VERSION_MAJOR
    #define OPTIM_VERSION_MAJOR 1
#endif

#ifndef OPTIM_VERSION_MINOR
    #define OPTIM_VERSION_MINOR 0
#endif

#ifndef OPTIM_VERSION_PATCH
    #define OPTIM_VERSION_PATCH 0
#endif

//

#if defined(_OPENMP) && !defined(OPTIM_NO_OMP) && !defined(OPTIM_USE_OPENMP)
    #define OPTIM_USE_OPENMP
#endif

//

#ifdef USE_RCPP_ARMADILLO
    #include <RcppArmadillo.h>
#else
    #ifndef ARMA_DONT_USE_WRAPPER
        #define ARMA_DONT_USE_WRAPPER
    #endif
    #include "armadillo"
#endif

namespace optim
{
    static const double inf = std::numeric_limits<double>::infinity();
    using uint_t = unsigned int;
}
