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
// this example is from Matlab's help page
// https://www.mathworks.com/help/optim/ug/fsolve.html
//
// F(x) = [exp(-exp(-(x_1+x_2))) - x_2*(1+x_1^2);
//         x_1*cos(x_2) + x_2*sin(x_1) - 0.5     ]
// 
// solution is: (0.3532,0.6061)
//

#ifndef _optim_zeros_test_fn_1_HPP
#define _optim_zeros_test_fn_1_HPP

arma::vec
zeros_test_objfn_1(const arma::vec& vals_inp, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    arma::vec ret(2);

    ret(0) = std::exp(-std::exp(-(x_1+x_2))) - x_2*(1 + std::pow(x_1,2));
    ret(1) = x_1*std::cos(x_2) + x_2*std::sin(x_1) - 0.5;
    //
    return ret;
}

arma::mat
zeros_test_jacob_1(const arma::vec& vals_inp, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    arma::mat ret(2,2);

    ret(0,0) = std::exp(-std::exp(-(x_1+x_2))-(x_1+x_2)) - 2*x_1*x_1;
    ret(0,1) = std::exp(-std::exp(-(x_1+x_2))-(x_1+x_2)) - x_1*x_1 - 1.0;
    ret(1,0) = std::cos(x_2) + x_2*std::cos(x_1);
    ret(1,1) = -x_1*std::sin(x_2) + std::cos(x_1);
    //
    return ret;
}

#endif
