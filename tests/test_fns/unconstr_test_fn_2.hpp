/*################################################################################
  ##
  ##   Copyright (C) 2016-2018 Keith O'Hara
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
// this example is from Matlab's help page
// https://www.mathworks.com/help/optim/ug/fminunc.html
//
// Rosenbrock's function:
// f(x) = 100*(x_2 - x_1^2)^2 + (1-x_1)^2
// 
// solution is: (1.00,1.00)
//

#ifndef _optim_test_fn_2_HPP
#define _optim_test_fn_2_HPP

double unconstr_test_fn_2(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data);

double
unconstr_test_fn_2(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    double obj_val = 100*std::pow(x_2 - std::pow(x_1,2),2) + std::pow(1-x_1,2);
    //
    if (grad_out) {
        (*grad_out)(0) = -400*(x_2 - std::pow(x_1,2))*x_1 - 2*(1-x_1);
        (*grad_out)(1) = 200*(x_2 - std::pow(x_1,2));
    }
    //
    return obj_val;
}

#endif
