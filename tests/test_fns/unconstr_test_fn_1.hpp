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
// this example is from Matlab's help page
// https://www.mathworks.com/help/optim/ug/fminunc.html
//
// f(x) = 3*x_1^2 + 2*x_1*x_2 + x_2^2 − 4*x_1 + 5*x_2
// 
// solution is: (2.25,-4.75)
//

#ifndef _optim_test_fn_1_HPP
#define _optim_test_fn_1_HPP

inline
double
unconstr_test_fn_1_whess(const ColVec_t& vals_inp, ColVec_t* grad_out, Mat_t* hess_out, void* opt_data)
{
    const double x_1 = vals_inp(0);
    const double x_2 = vals_inp(1);

    double obj_val = 3*x_1*x_1 + 2*x_1*x_2 + x_2*x_2 - 4*x_1 + 5*x_2;

    if (grad_out) {
        (*grad_out)(0) = 6*x_1 + 2*x_2 - 4;
        (*grad_out)(1) = 2*x_1 + 2*x_2 + 5;
    }

    if (hess_out) {
        (*hess_out)(0,0) = 6.0;
        (*hess_out)(0,1) = 2.0;
        (*hess_out)(1,0) = 2.0;
        (*hess_out)(1,1) = 2.0;
    }

    //
    
    return obj_val;
}

inline
double
unconstr_test_fn_1(const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)
{
    return unconstr_test_fn_1_whess(vals_inp, grad_out, nullptr, opt_data);
}

#endif
