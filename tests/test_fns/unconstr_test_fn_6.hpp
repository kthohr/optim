/*################################################################################
  ##
  ##   Copyright (C) 2016-2023 Keith O'Hara
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
// Rastrigin function:
//
// f(x) = A*n + sum_{i=1}^n (x_i^2 - A cos(2*pi*x_i))
// where A = 10
// 
// solution is: (0,0)
//

#ifndef _optim_test_fn_6_HPP
#define _optim_test_fn_6_HPP

struct unconstr_test_fn_6_data {
    double A;
};

inline
double
unconstr_test_fn_6(const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)
{
    const int n = BMO_MATOPS_SIZE(vals_inp);

    unconstr_test_fn_6_data* objfn_data = reinterpret_cast<unconstr_test_fn_6_data*>(opt_data);
    const double A = objfn_data->A;

    double obj_val = A*n + BMO_MATOPS_ACCU( BMO_MATOPS_POW(vals_inp,2) - A*BMO_MATOPS_COS(2 * OPTIM_PI * vals_inp) );
    //
    // if (grad_out) {
    //     *grad_out = 2*vals_inp + A*2*arma::datum::pi*arma::sin(2*arma::datum::pi*vals_inp);
    // }
    //
    return obj_val;
}

#endif
