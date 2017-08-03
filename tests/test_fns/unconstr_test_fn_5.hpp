/*################################################################################
  ##
  ##   Copyright (C) 2016-2017 Keith O'Hara
  ##
  ##   This file is part of the OptimLib C++ library.
  ##
  ##   OptimLib is free software: you can redistribute it and/or modify
  ##   it under the terms of the GNU General Public License as published by
  ##   the Free Software Foundation, either version 2 of the License, or
  ##   (at your option) any later version.
  ##
  ##   OptimLib is distributed in the hope that it will be useful,
  ##   but WITHOUT ANY WARRANTY; without even the implied warranty of
  ##   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  ##   GNU General Public License for more details.
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
// solution is: (1,3)
//

#ifndef _optim_test_fn_5_HPP
#define _optim_test_fn_5_HPP

double unconstr_test_fn_5(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data);

double
unconstr_test_fn_5(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    double obj_val = std::pow(x_1 + 2*x_2 - 7.0,2) + std::pow(2*x_1 + x_2 - 5.0,2);
    //
    if (grad_out) {
        (*grad_out)(0) = 2*(x_1 + 2*x_2 - 7.0) + 2*(2*x_1 + x_2 - 5.0)*2;
        (*grad_out)(1) = 2*(x_1 + 2*x_2 - 7.0)*2 + 2*(2*x_1 + x_2 - 5.0);
    }
    //
    return obj_val;
}

#endif
