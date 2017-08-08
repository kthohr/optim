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
// Table function:
//
// f(x,y) = -abs( sin(x)cos(y)exp( abs(1 - sqrt(x^2 + y^2)/pi) ) )
// -10 <= x,y <= 10
//
// there are four solutions: (8.05502,9.66459), (-8.05502,9.66459), (8.05502,-9.66459), (-8.05502,-9.66459)
//

#ifndef _optim_test_fn_10_HPP
#define _optim_test_fn_10_HPP

double unconstr_test_fn_10(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data);

double 
unconstr_test_fn_10(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
{
    const double x = vals_inp(0);
    const double y = vals_inp(1);
    const double pi = arma::datum::pi;

    double obj_val = - std::abs( std::sin(x)*std::cos(y)*std::exp( std::abs(1.0 - std::sqrt(x*x + y*y)/pi) ) );
    //
    return obj_val;
}

#endif
