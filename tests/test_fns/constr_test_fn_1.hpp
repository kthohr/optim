/*################################################################################
  ##
  ##   Copyright (C) 2016-2023 Keith O'Hara
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
// g(x) = -2*x_1 - x_2 + 14 <= 0
// 
// solution is: (5,4)
//

#ifndef _optim_constr_test_fn_1_HPP
#define _optim_constr_test_fn_1_HPP

double
constr_test_objfn_1(const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)
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

ColVec_t
constr_test_constrfn_1(const ColVec_t& vals_inp, Mat_t* jacob_out, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    ColVec_t constr_vals(1);
    constr_vals(0) = -2*x_1 - x_2 + 14.0;
    
    if (jacob_out) {
        BMO_MATOPS_SET_SIZE_POINTER(jacob_out,1,2);

        (*jacob_out)(0,0) = -2.0;
        (*jacob_out)(0,1) = -1.0;
    }
    
    return constr_vals;
}

#endif
