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

/*
 * Newton test - logit regression
 */

#include "optim.hpp"

using optim::ColVec_t;
using optim::Mat_t;

// sigmoid function

inline
Mat_t 
sigm(const Mat_t& X)
{
    return BMO_MATOPS_SCALAR_DIV_ARRAY(1.0, ( BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_EXP(-X), 1.0) ));
}

// log-likelihood function

struct ll_data_t {
    ColVec_t Y;
    Mat_t X;
};

double ll_fn(const ColVec_t& vals_inp, ColVec_t* grad_out, Mat_t* hess_out, void* opt_data)
{
    ll_data_t* objfn_data = reinterpret_cast<ll_data_t*>(opt_data);

    ColVec_t Y = objfn_data->Y;
    Mat_t X = objfn_data->X;

    ColVec_t mu = sigm(X*vals_inp);

    double norm_term = static_cast<double>( BMO_MATOPS_SIZE(Y) );

#ifndef OPTIM_LOGIT_EX_LL_TERM_1
    #define OPTIM_LOGIT_EX_LL_TERM_1 BMO_MATOPS_HADAMARD_PROD(Y, BMO_MATOPS_LOG(mu))
#endif

#ifndef OPTIM_LOGIT_EX_LL_TERM_2
    #define OPTIM_LOGIT_EX_LL_TERM_2 BMO_MATOPS_HADAMARD_PROD( BMO_MATOPS_ARRAY_ADD_SCALAR(-Y,1.0), BMO_MATOPS_LOG( BMO_MATOPS_ARRAY_ADD_SCALAR(-mu,1.0) ))
#endif

    const double obj_val = - BMO_MATOPS_ACCU( OPTIM_LOGIT_EX_LL_TERM_1 + OPTIM_LOGIT_EX_LL_TERM_2 ) / norm_term; 

    //

    if (grad_out) {
        *grad_out = BMO_MATOPS_TRANSPOSE(X) * (mu - Y) / norm_term;
    }

    if (hess_out) {
        Mat_t S = BMO_MATOPS_DIAGMAT( BMO_MATOPS_HADAMARD_PROD(mu, BMO_MATOPS_ARRAY_ADD_SCALAR(-mu,1.0)) );
        *hess_out = BMO_MATOPS_TRANSPOSE(X) * S * X / norm_term;
    }

    //

    return obj_val;
}

//

int main()
{
    int n_dim = 5;     // dimension of theta
    int n_samp = 1000; // sample length

    Mat_t X = optim::bmo_stats::rsnorm_mat<optim::fp_t>(n_samp,n_dim);
    ColVec_t theta_0 = BMO_MATOPS_ARRAY_ADD_SCALAR(3.0 * BMO_MATOPS_RANDU_VEC(n_dim), 1.0);

    BMO_MATOPS_COUT << "\nTrue theta:\n" << theta_0 << "\n";

    ColVec_t mu = sigm(X*theta_0);

    ColVec_t Y(n_samp);

    for (int i = 0; i < n_samp; ++i) {
        Y(i) = ( BMO_MATOPS_AS_SCALAR(BMO_MATOPS_RANDU_VEC(1)) < mu(i) ) ? 1.0 : 0.0;
    }

    // fn data and initial values

    ll_data_t opt_data;
    opt_data.Y = std::move(Y);
    opt_data.X = std::move(X);

    ColVec_t x = BMO_MATOPS_ARRAY_ADD_SCALAR( BMO_MATOPS_ONE_COLVEC(n_dim), 1.0 ); // (2,2)

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
 
    bool success = optim::newton(x, ll_fn, &opt_data);

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
 
    if (success) {
        std::cout << "newton: logit_reg test completed successfully.\n"
                  << "elapsed time: " << elapsed_seconds.count() << "s\n";
    } else {
        std::cout << "newton: logit_reg test completed unsuccessfully." << std::endl;
    }
 
    BMO_MATOPS_COUT << "\nnewton: solution to logit_reg test:\n" << x << "\n";
 
    return 0;
}