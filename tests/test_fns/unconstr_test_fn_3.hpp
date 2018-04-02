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

//
// this example is from
// https://en.wikipedia.org/wiki/Test_functions_for_optimization
//
// Sphere function:
//
// f(x) = x_1^2 + x_2^2 + ... + x_n^2
// 
// solution is: (0,0,...,0)
//

#ifndef _optim_test_fn_3_HPP
#define _optim_test_fn_3_HPP

double unconstr_test_fn_3(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data);
double unconstr_test_fn_3_whess(const arma::vec& vals_inp, arma::vec* grad_out, arma::mat* hess_out, void* opt_data);

double
unconstr_test_fn_3(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
{
    const double obj_val = arma::dot(vals_inp,vals_inp);

    //

    if (grad_out) {
        *grad_out = 2.0*vals_inp;
    }

    //

    return obj_val;
}

double
unconstr_test_fn_3_whess(const arma::vec& vals_inp, arma::vec* grad_out, arma::mat* hess_out, void* opt_data)
{
    const int n_vals = vals_inp.n_elem;
    const double obj_val = arma::dot(vals_inp,vals_inp);

    //

    if (grad_out) {
        *grad_out = 2.0*vals_inp;
    }

    if (hess_out) {
        *hess_out = 2.0*arma::eye(n_vals,n_vals);
    }

    //

    return obj_val;
}

#endif
