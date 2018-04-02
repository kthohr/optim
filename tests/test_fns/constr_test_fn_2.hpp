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
// simple constrained optim problem
// 
// f(x) = (x_1 - 5)^2 + (x_2 - 4)^2
// g_1(x) = -2*x_1 - x_2 + 14 <= 0
// g_2(x) = x_1 + x_2 - 9 <= 0
// 
// solution is: (5,4)
//

#ifndef _optim_constr_test_fn_2_HPP
#define _optim_constr_test_fn_2_HPP

double
constr_test_objfn_2(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    double obj_val = std::pow(x_1 - 5.0,2) + std::pow(x_2 - 4.0,2);
    //
    if (grad_out) {
        (*grad_out)(0) = 2.0*(x_1 - 5.0);
        (*grad_out)(1) = 2.0*(x_2 - 4.0);
    }
    //
    return obj_val;
}

arma::vec
constr_test_constrfn_2(const arma::vec& vals_inp, arma::mat* jacob_out, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    arma::vec constr_vals(2);
    constr_vals(0) = -2*x_1 - x_2 + 14.0;
    constr_vals(1) = x_1 + x_2 - 9.0;
    //
    if (jacob_out) {
        jacob_out->set_size(2,2);

        (*jacob_out)(0,0) = -2.0;
        (*jacob_out)(0,1) = -1.0;
        (*jacob_out)(1,0) = 1.0;
        (*jacob_out)(1,1) = 1.0;
    }
    //
    return constr_vals;
}

#endif
