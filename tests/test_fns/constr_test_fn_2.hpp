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
// simple constrained optim problem
// 
// f(x) = (x_1 - 5)^2 + (x_2 - 4)^2
// g_1(x) = -2*x_1 - x_2 + 14 <= 0
// g_2(x) = x_1 + x_2 - 9 <= 0
// 
// solution is: (5,4)
//

#ifndef _optim_constr_test_fn_2_HPP
#define _optim_constr_test_fn_2_HPP

double
constr_test_objfn_2(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
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

arma::vec
constr_test_constrfn_2(const arma::vec& vals_inp, arma::mat* jacob_out, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    arma::vec constr_vals(2);
    constr_vals(0) = -2*x_1 - x_2 + 14.0;
    constr_vals(1) = x_1 + x_2 - 9.0;
    //
    if (jacob_out) {
        jacob_out->set_size(2,2);

        (*jacob_out)(0,0) = -2.0;
        (*jacob_out)(0,1) = -1.0;
        (*jacob_out)(1,0) = 1.0;
        (*jacob_out)(1,1) = 1.0;
    }
    //
    return constr_vals;
}

#endif
