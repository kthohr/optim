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

/*
 * zeros
 */

#ifndef OPTIM_MATOPS_ONES

//

#ifdef OPTIM_ENABLE_ARMA_WRAPPERS
    #define OPTIM_MATOPS_ONE_VEC(n) arma::ones(n,1)
    #define OPTIM_MATOPS_ONE_MAT(n,k) arma::ones(n,k)
#endif

#ifdef OPTIM_ENABLE_EIGEN_WRAPPERS
    // #define OPTIM_MATOPS_ONE_VEC(n) Eigen::ArrayXd::Ones(n)
    #define OPTIM_MATOPS_ONE_VEC(n) Eigen::VectorXd::Ones(n)
    // #define OPTIM_MATOPS_ONE_MAT(n,k) Eigen::ArrayXXd::Ones(n,k)
    #define OPTIM_MATOPS_ONE_MAT(n,k) Eigen::MatrixXd::Ones(n,k)
#endif

//

#endif
