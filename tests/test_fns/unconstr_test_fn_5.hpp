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

//
// this example is from
// https://en.wikipedia.org/wiki/Test_functions_for_optimization
//
// Booth's function:
//
// f(x) = (x_1 + 2*x_2 - 7)^2 + (2*x_1 + x_2 - 5)^2
// s.t. -10 <= x_1, x_2 <= 10
// 
// solution: f(1,3) = 0
//

#ifndef _optim_test_fn_5_HPP
#define _optim_test_fn_5_HPP

inline
double
unconstr_test_fn_5_whess(const Vec_t& vals_inp, Vec_t* grad_out, Mat_t* hess_out, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    double obj_val = std::pow(x_1 + 2*x_2 - 7.0, 2) + std::pow(2*x_1 + x_2 - 5.0, 2);
    
    if (grad_out) {
        // (*grad_out)(0) = 2*(x_1 + 2*x_2 - 7.0) + 2*(2*x_1 + x_2 - 5.0)*2;
        (*grad_out)(0) = 10*x_1 + 8*x_2 - 34;
        // (*grad_out)(1) = 2*(x_1 + 2*x_2 - 7.0)*2 + 2*(2*x_1 + x_2 - 5.0);
        (*grad_out)(1) = 8*x_1 + 10*x_2 - 38;
    }

    if (hess_out) {
        (*hess_out)(0,0) = 10.0;
        (*hess_out)(0,1) = 8.0;
        (*hess_out)(1,0) = 8.0;
        (*hess_out)(1,1) = 10.0;
    }
    
    return obj_val;
}

inline
double
unconstr_test_fn_5(const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)
{
    return unconstr_test_fn_5_whess(vals_inp, grad_out, nullptr, opt_data);
}

#endif
