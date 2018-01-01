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
// this example is from
// https://en.wikipedia.org/wiki/Test_functions_for_optimization
//
// Rosenbrock function constrained with a cubic and a line
// 
// f(x) = (1 - x)^2 + 100(y - x^2)^2
// g_1(x) = (x - 1)^3 - y + 1 <= 0
// g_2(x) = x + y - 2 <= 0
// 
// solution is: (1,1)
//

#ifndef _optim_constr_test_fn_3_HPP
#define _optim_constr_test_fn_3_HPP

double
constr_test_objfn_3(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
{
    double x = vals_inp(0);
    double y = vals_inp(1);

    double obj_val = std::pow(1.0 - x,2) + 100*std::pow(y - x*x,2);
    //
    if (grad_out) {
        (*grad_out)(0) = - 2*(1.0 - x) - 200*(y - x*x)*2*x;
        (*grad_out)(1) = 200*(y - x*x);
    }
    //
    return obj_val;
}

arma::vec
constr_test_constrfn_3(const arma::vec& vals_inp, arma::mat* jacob_out, void* opt_data)
{
    double x = vals_inp(0);
    double y = vals_inp(1);

    arma::vec constr_vals(2);
    constr_vals(0) = std::pow(x - 1.0,3) - y + 1.0;
    constr_vals(1) = x + y - 2.0;
    //
    if (jacob_out) {
        jacob_out->set_size(2,2);

        (*jacob_out)(0,0) = 3*std::pow(x - 1.0,2);
        (*jacob_out)(0,1) = -1.0;
        (*jacob_out)(1,0) = 1.0;
        (*jacob_out)(1,1) = 1.0;
    }
    //
    return constr_vals;
}

#endif
