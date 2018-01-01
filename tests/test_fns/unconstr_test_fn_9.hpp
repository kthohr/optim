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
// Bukin function N.6:
//
// f(x) = 100*sqrt(abs(y - 0.01*x^2)) + 0.01*abs(x + 10)
// -15 <= x <= -5
// - 3 <= y <= 3
//
// solution is: (-10,1)
//

#ifndef _optim_test_fn_9_HPP
#define _optim_test_fn_9_HPP

double unconstr_test_fn_9(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data);

double 
unconstr_test_fn_9(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
{
    const double x = vals_inp(0);
    const double y = vals_inp(1);

    double obj_val = 100*std::sqrt(std::abs(y - 0.01*x*x)) + 0.01*std::abs(x + 10);
    //
    return obj_val;
}

#endif
