/*################################################################################
  ##
  ##   Copyright (C) 2016-2018 Keith O'Hara
  ##
  ##   This file is part of the OptimLib C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
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
