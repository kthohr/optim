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
