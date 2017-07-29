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
// Sphere function:
//
// f(x) = x_1^2 + x_2^2 + ... + x_n^2
// 
// solution is: (0,0,...,0)
//

#ifndef _optim_test_fn_3_HPP
#define _optim_test_fn_3_HPP

double unconstr_test_fn_3(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data);

double
unconstr_test_fn_3(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
{
    double obj_val = arma::dot(vals_inp,vals_inp);
    //
    if (grad) {
        *grad = 2.0*vals_inp;
    }
    //
    return obj_val;
}

#endif
