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
// Beale's function:
//
// f(x) = (1.5 - x_1 + x_1*x_2)^2 + (2.25 - x_1 + x_1*x_2^2)^2 + (2.625 - x_1 + x_1*x_2^3)^2
// -4.5 <= x,y <= 4.5
//
// solution is: (3.0,0.5)
//

#ifndef _optim_test_fn_4_HPP
#define _optim_test_fn_4_HPP

double unconstr_test_fn_4(const arma::vec& vals_inp, arma::vec* grad, void* opt_data);

double 
unconstr_test_fn_4(const arma::vec& vals_inp, arma::vec* grad, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    double obj_val = std::pow(1.5 - x_1 + x_1*x_2,2) + std::pow(2.25 - x_1 + x_1*std::pow(x_2,2),2) + std::pow(2.625 - x_1 + x_1*std::pow(x_2,3),2);
    //
    if (grad) {
        (*grad)(0) = 2*(1.5 - x_1 + x_1*x_2)*(-1 + x_2) + 2*(2.25 - x_1 + x_1*std::pow(x_2,2))*(-1 + std::pow(x_2,2)) + 2*(2.625 - x_1 + x_1*std::pow(x_2,3))*(- 1 + std::pow(x_2,3));
        (*grad)(1) = 2*(1.5 - x_1 + x_1*x_2)*(x_1) + 2*(2.25 - x_1 + x_1*std::pow(x_2,2))*(x_1*2*x_2) + 2*(2.625 - x_1 + x_1*std::pow(x_2,3))*(x_1*3*std::pow(x_2,2));
    }
    //
    return obj_val;
}

#endif
