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

#ifndef OPTIM_MATOPS_RANDN

//

#ifdef OPTIM_ENABLE_ARMA_WRAPPERS
    #define OPTIM_MATOPS_RANDN_VEC(j) arma::randn(j,1)
    #define OPTIM_MATOPS_RANDN_ROWVEC(j) arma::randn(1,j)
    #define OPTIM_MATOPS_RANDN_MAT(j,k) arma::randn(j,k)
#endif

#ifdef OPTIM_ENABLE_EIGEN_WRAPPERS
    inline
    Vec_t
    eigen_randn_vec(size_t nr)
    {
        static std::mt19937 gen{ std::random_device{}() };
        static std::normal_distribution<> dist;

        return Eigen::VectorXd{ nr }.unaryExpr([&](double x) { return dist(gen); });
    }

    inline
    Mat_t
    eigen_randn_mat(size_t nr, size_t nc)
    {
        static std::mt19937 gen{ std::random_device{}() };
        static std::normal_distribution<> dist;

        return Eigen::MatrixXd{ nr, nc }.unaryExpr([&](double x) { return dist(gen); });
    }

    #define OPTIM_MATOPS_RANDN_VEC(j) optim::eigen_randn_vec(j)
    #define OPTIM_MATOPS_RANDN_ROWVEC(j) (optim::eigen_randn_vec(j)).transpose()
    #define OPTIM_MATOPS_RANDN_MAT(j,k) optim::eigen_randn_mat(j,k)
#endif

//

#endif
