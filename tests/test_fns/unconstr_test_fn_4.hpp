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
// this example is from
// https://en.wikipedia.org/wiki/Test_functions_for_optimization
//
// Beale's function:
//
// f(x) = (1.5 - x_1 + x_1*x_2)^2 + (2.25 - x_1 + x_1*x_2^2)^2 + (2.625 - x_1 + x_1*x_2^3)^2
// -4.5 <= x_1, x_2 <= 4.5
//
// solution is: (3.0, 0.5)
//

#ifndef _optim_test_fn_4_HPP
#define _optim_test_fn_4_HPP

inline
double 
unconstr_test_fn_4_whess(const ColVec_t& vals_inp, ColVec_t* grad_out, Mat_t* hess_out, void* opt_data)
{
    const double x_1 = vals_inp(0);
    const double x_2 = vals_inp(1);

    // compute some terms only once 

    const double x1sq = x_1 * x_1;
    const double x2sq = x_2 * x_2;
    const double x2cb = x2sq * x_2;
    const double x1x2 = x_1*x_2;

    // 

    double obj_val = std::pow(1.5 - x_1 + x1x2, 2) + std::pow(2.25 - x_1 + x_1*x2sq, 2) + std::pow(2.625 - x_1 + x_1*x2cb, 2);

    if (grad_out) {
        //                2 x_1 x_2^6            +  2 x_1 x_2^4            -  4 x_1 x_2^3     -  2 x_1 x_2^2     -  4 x_1 x_2 +  6 x_1    +  5.25 x_2^3   +  4.5 x_2^2   +  3 x_2    - 12.75
        (*grad_out)(0) = (2 * x_1 * x2cb * x2cb) + (2 * x_1 * x2sq * x2sq) - (4 * x_1 * x2cb) - (2 * x_1 * x2sq) - (4 * x1x2) + (6 * x_1) + (5.25 * x2cb) + (4.5 * x2sq) + (3 * x_2) - 12.75;
        //                6 x_1^2 x_2^5           +  4 x_1^2 x_2^3    -  6 x_1^2 x_2^2    -  2 x_1^2 x_2     -  2 x_1^2   +  15.75 x_1 x_2^2     +  9 x_1 x_2 +  3 x_1
        (*grad_out)(1) = (6 * x1sq * x2sq * x2cb) + (4 * x1sq * x2cb) - (6 * x1sq * x2sq) - (2 * x1sq * x_2) - (2 * x1sq) + (15.75 * x_1 * x2sq) + (9 * x1x2) + (3 * x_1);
    }

    if (hess_out) {
        //                  2 x_2^6          +  2 x_2^4          -  4 x_2^3 -  2 x_2^2 -  4 x_2  + 6
        (*hess_out)(0,0) = (2 * x2cb * x2cb) + (2 * x2sq * x2sq) - (4*x2cb) - (2*x2sq) - (4*x_2) + 6;

        //            12 x_1 x_2^5            +  8 x_1 x_2^3     -  12 x_1 x_2^2     -  4 x_1 x_2 -  4 x_1    +  15.75 x_2^2   +  9 x_2    + 3
        double H12 = (12 * x_1 * x2sq * x2cb) + (8 * x_1 * x2cb) - (12 * x_1 * x2sq) - (4 * x1x2) - (4 * x_1) + (15.75 * x2sq) + (9 * x_2) + 3;
        (*hess_out)(0,1) = H12;
        (*hess_out)(1,0) = H12;

        //                  30 x_1^2 x_2^4           +  12 x_1^2 x_2^2    -  12 x_1^2 x_2     -  2 x_1^2 +  31.5 x_1 x_2 +  9 x_1
        (*hess_out)(1,1) = (30 * x1sq * (x2sq*x2sq)) + (12 * x1sq * x2sq) - (12 * x1sq * x_2) - (2*x1sq) + (31.5 * x1x2) + (9 * x_1);
    }

    return obj_val;
}

inline
double
unconstr_test_fn_4(const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)
{
    return unconstr_test_fn_4_whess(vals_inp, grad_out, nullptr, opt_data);
}

#endif
