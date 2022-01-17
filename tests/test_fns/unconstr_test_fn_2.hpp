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

//
// Rosenbrock's function:
// f(x) = 100*(x_2 - x_1^2)^2 + (1-x_1)^2
// 
// solution is: (1.00,1.00)
//

#ifndef _optim_test_fn_2_HPP
#define _optim_test_fn_2_HPP

inline
double
unconstr_test_fn_2_whess(const ColVec_t& vals_inp, ColVec_t* grad_out, Mat_t* hess_out, void* opt_data)
{
    const double x_1 = vals_inp(0);
    const double x_2 = vals_inp(1);

    const double x1sq = x_1 * x_1;

    //

    double obj_val = 100*std::pow(x_2 - x1sq,2) + std::pow(1-x_1,2);
    
    if (grad_out) {
        (*grad_out)(0) = -400*(x_2 - x1sq)*x_1 - 2*(1-x_1);
        (*grad_out)(1) = 200*(x_2 - x1sq);
    }

    if (hess_out) {
        (*hess_out)(0,0) = 1200 * x1sq - 400 * x_2 + 2;
        (*hess_out)(0,1) = - 400 * x_1;
        (*hess_out)(1,0) = - 400 * x_1;
        (*hess_out)(1,1) = 200;
    }
    
    return obj_val;
}

inline
double
unconstr_test_fn_2(const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)
{
    return unconstr_test_fn_2_whess(vals_inp, grad_out, nullptr, opt_data);
}

#endif
