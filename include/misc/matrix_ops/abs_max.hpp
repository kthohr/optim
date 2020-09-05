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

#ifndef OPTIM_MATOPS_ABS_MAX

//

#ifdef OPTIM_ENABLE_ARMA_WRAPPERS
    #define OPTIM_MATOPS_ABS_MAX(x,y) arma::max(arma::abs(x),arma::abs(y))

    #define OPTIM_MATOPS_ABS_MAX_VAL(x) (arma::abs(x)).max()
    #define OPTIM_MATOPS_COLWISE_ABS_MAX(x) arma::max(arma::abs(x),0)
    #define OPTIM_MATOPS_ROWWISE_ABS_MAX(x) arma::max(arma::abs(x),1)
#endif

#ifdef OPTIM_ENABLE_EIGEN_WRAPPERS
    // #define OPTIM_MATOPS_MAX(x,y) x.cwiseMax(y)
    #define OPTIM_MATOPS_ABS_MAX(x,y) (x).cwiseAbs().array().max((y).cwiseAbs().array())

    #define OPTIM_MATOPS_ABS_MAX_VAL(x) (x).cwiseAbs().maxCoeff()
    #define OPTIM_MATOPS_COLWISE_ABS_MAX(x) (x).cwiseAbs().colwise().maxCoeff()
    #define OPTIM_MATOPS_ROWWISE_ABS_MAX(x) (x).cwiseAbs().rowwise().maxCoeff()
#endif

//

#endif
