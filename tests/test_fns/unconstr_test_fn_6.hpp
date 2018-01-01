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
// Rastrigin function:
//
// f(x) = A*n + sum_{i=1}^n (x_i^2 - A cos(2*pi*x_i))
// where A = 10
// 
// solution is: (0,0)
//

#ifndef _optim_test_fn_6_HPP
#define _optim_test_fn_6_HPP

struct unconstr_test_fn_6_data {
    double A;
};

double unconstr_test_fn_6(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data);

double
unconstr_test_fn_6(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
{
    const int n = vals_inp.n_elem;

    unconstr_test_fn_6_data* objfn_data = reinterpret_cast<unconstr_test_fn_6_data*>(opt_data);
    const double A = objfn_data->A;

    double obj_val = A*n + arma::accu( arma::pow(vals_inp,2) - A*arma::cos(2*arma::datum::pi*vals_inp) );
    //
    // if (grad_out) {
    //     *grad_out = 2*vals_inp + A*2*arma::datum::pi*arma::sin(2*arma::datum::pi*vals_inp);
    // }
    //
    return obj_val;
}

#endif
