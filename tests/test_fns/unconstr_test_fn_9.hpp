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
