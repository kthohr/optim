/*################################################################################
  ##
  ##   Copyright (C) 2016-2020 Keith O'Hara
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

#ifndef OPTIM_MATOPS_ACCESS

//

#ifdef OPTIM_ENABLE_ARMA_WRAPPERS
    #define OPTIM_MATOPS_COLS(x, v) (x).cols(v) // v is a vector
    #define OPTIM_MATOPS_ROWS(x, v) (x).rows(v) // v is a vector
    // acces columns j through k
    #define OPTIM_MATOPS_MIDDLE_COLS(x, j, k) (x).cols(j,k)
    #define OPTIM_MATOPS_MIDDLE_ROWS(x, j, k) (x).rows(j,k)
#endif

#ifdef OPTIM_ENABLE_EIGEN_WRAPPERS
    #define OPTIM_MATOPS_COLS(x, v) (x)(Eigen::all,v) // v is a vector
    #define OPTIM_MATOPS_ROWS(x, v) (x)(v,Eigen::all) // v is a vector
    // acces columns j through k
    #define OPTIM_MATOPS_MIDDLE_COLS(x, j, k) (x).middleCols(j,k-j+1)
    #define OPTIM_MATOPS_MIDDLE_ROWS(x, j, k) (x).middleRows(j,k-j+1)
#endif

//

#endif
