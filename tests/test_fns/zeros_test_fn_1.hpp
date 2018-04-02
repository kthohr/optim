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
