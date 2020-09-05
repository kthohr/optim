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
 * Lp-Norm
 */

#ifndef OPTIM_MATOPS_NORM

//

#ifdef OPTIM_ENABLE_ARMA_WRAPPERS
    #define OPTIM_MATOPS_L1NORM(x) arma::norm(x,1)
    #define OPTIM_MATOPS_L2NORM(x) arma::norm(x,2)
    #define OPTIM_MATOPS_LPNORM(x,p) arma::norm(x,p)
    #define OPTIM_MATOPS_LINFNORM(x) arma::norm(x,"inf")
    #define OPTIM_MATOPS_LMINFNORM(x) arma::norm(x,"-inf")
#endif

#ifdef OPTIM_ENABLE_EIGEN_WRAPPERS
    #define OPTIM_MATOPS_L1NORM(x) (x).array().abs().sum()
    #define OPTIM_MATOPS_L2NORM(x) (x).norm()
    // #define OPTIM_MATOPS_LPNORM(x,p) x.lpNorm<p>()
    #define OPTIM_MATOPS_LPNORM(x,p) std::pow( (x).array().abs().pow(p).sum() , 1.0 / static_cast<double>(p) )
    #define OPTIM_MATOPS_LINFNORM(x) (x).lpNorm<Eigen::Infinity>()
    #define OPTIM_MATOPS_LMINFNORM(x) (x).cwiseAbs().minCoeff()
#endif

//

#endif
