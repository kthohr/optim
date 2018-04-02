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

/*
 * Newton test - logit regression
 */

#include "optim.hpp"

// sigmoid function

arma::mat sigm(const arma::mat& X)
{
    return 1.0 / (1.0 + arma::exp(-X));
}

// log-likelihood function

struct ll_data_t {
    arma::vec Y;
    arma::mat X;
};

double ll_fn(const arma::vec& vals_inp, arma::vec* grad_out, arma::mat* hess_out, void* opt_data)
{
    ll_data_t* objfn_data = reinterpret_cast<ll_data_t*>(opt_data);

    arma::vec Y = objfn_data->Y;
    arma::mat X = objfn_data->X;

    arma::vec mu = sigm(X*vals_inp);

    double norm_term = static_cast<double>(Y.n_elem);

    const double obj_val = - arma::accu( Y%arma::log(mu) + (1.0-Y)%arma::log(1.0-mu) ) / norm_term;

    //

    if (grad_out)
    {
        *grad_out = X.t() * (mu - Y) / norm_term;
    }

    if (hess_out)
    {
        arma::mat S = arma::diagmat( mu%(1.0-mu) );
        *hess_out = X.t() * S * X / norm_term;
    }

    //

    return obj_val;
}

//

int main()
{
    int n_dim = 5;     // dimension of theta
    int n_samp = 1000; // sample length

    arma::mat X = arma::randn(n_samp,n_dim);
    arma::vec theta_0 = 1.0 + 3.0*arma::randu(n_dim,1);

    arma::cout << "\nTrue theta:\n" << theta_0 << arma::endl;

    arma::vec mu = sigm(X*theta_0);

    arma::vec Y(n_samp);

    for (int i=0; i < n_samp; i++)
    {
        Y(i) = ( arma::as_scalar(arma::randu(1)) < mu(i) ) ? 1.0 : 0.0;
    }

    // fn data and initial values

    ll_data_t opt_data;
    opt_data.Y = std::move(Y);
    opt_data.X = std::move(X);

    arma::vec x = arma::ones(n_dim,1) + 1.0; // (2,2)

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
 
    bool success = optim::newton(x,ll_fn,&opt_data);

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
 
    if (success) {
        std::cout << "newton: logit_reg test completed successfully.\n"
                  << "elapsed time: " << elapsed_seconds.count() << "s\n";
    } else {
        std::cout << "newton: logit_reg test completed unsuccessfully." << std::endl;
    }
 
    arma::cout << "\nnewton: solution to logit_reg test:\n" << x << arma::endl;
 
    return 0;
}