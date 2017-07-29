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
// simple constrained optim problem
// 
// f(x) = (x_1 - 5)^2 + (x_2 - 4)^2
// g(x) = -2*x_1 - x_2 + 12 <= 0
// 
// solution is: (5,4)
//

#ifndef _optim_constr_test_fn_1_HPP
#define _optim_constr_test_fn_1_HPP

double
constr_test_objfn_1(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    double obj_val = std::pow(x_1 - 5.0,2) + std::pow(x_2 - 4.0,2);
    //
    if (grad_out) {
        (*grad_out)(0) = 2.0*(x_1 - 5.0);
        (*grad_out)(1) = 2.0*(x_2 - 4.0);
    }
    //
    return obj_val;
}

double
constr_test_constrfn_1(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    double constr_val = -2*x_1 - x_2 + 14.0;
    //
    if (grad_out) {
        (*grad_out)(0) = -2.0;
        (*grad_out)(1) = -1.0;
    }
    //
    return constr_val;
}

#endif
