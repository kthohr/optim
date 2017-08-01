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
// Levi function:
//
// f(x) = (sin(3*pi*x))^2 + (x-1)^2 (1 + (sin(3*pi*y))^2) + (y-1)^2 (1 + (sin(2*pi*x))^2)
// 
// solution is: (1,1)
//

#ifndef _optim_test_fn_8_HPP
#define _optim_test_fn_8_HPP

double unconstr_test_fn_8(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data);

double
unconstr_test_fn_8(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
{
    const double x = vals_inp(0);
    const double y = vals_inp(1);
    const double pi = arma::datum::pi;

    double obj_val = std::pow( std::sin(3*pi*x), 2) + std::pow(x-1,2)*(1 + std::pow( std::sin(3*pi*y), 2)) + std::pow(y-1,2)*(1 + std::pow( std::sin(2*pi*x), 2));
    //
    return obj_val;
}

#endif
