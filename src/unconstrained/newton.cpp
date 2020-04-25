/*################################################################################
  ##
  ##   Copyright (C) 2016-2020 Keith O'Hara
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

/*
 * Newton's method for non-linear optimization
 */

#include "optim.hpp"

// [OPTIM_BEGIN]
optimlib_inline
bool
optim::newton_int(Vec_t& init_out_vals, 
                  std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, Mat_t* hess_out, void* opt_data)> opt_objfn, 
                  void* opt_data, 
                  algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.

    bool success = false;

    const size_t n_vals = OPTIM_MATOPS_SIZE(init_out_vals);

    //
    // Newton settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }
    
    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const uint_t iter_max = settings.iter_max;
    const double err_tol = settings.err_tol;

    //
    // initialization

    Vec_t x = init_out_vals;

    if (! OPTIM_MATOPS_IS_FINITE(x) ) {
        printf("newton error: non-finite initial value(s).\n");
        return false;
    }

    Mat_t H(n_vals, n_vals); // hessian matrix
    Vec_t grad(n_vals);     // gradient vector

    opt_objfn(x, &grad, &H, opt_data);

    double err = OPTIM_MATOPS_L2NORM(grad);
    if (err <= err_tol) {
        return true;
    }

    //
    // if ||gradient(initial values)|| > tolerance, then continue

    Vec_t d = - OPTIM_MATOPS_SOLVE(H, grad); // Newton direction

    Vec_t x_p = x + d; // no line search used here

    opt_objfn(x_p,&grad,&H,opt_data);

    err = OPTIM_MATOPS_L2NORM(grad);
    if (err <= err_tol) {
        init_out_vals = x_p;
        return true;
    }

    //
    // begin loop

    uint_t iter = 0;

    while (err > err_tol && iter < iter_max) {
        iter++;

        //

        d = - OPTIM_MATOPS_SOLVE(H,grad);
        x_p = x + d;
        
        opt_objfn(x_p, &grad, &H, opt_data);
        
        //

        err = OPTIM_MATOPS_L2NORM(grad);
        if (err <= err_tol) {
            break;
        }

        err = OPTIM_MATOPS_L2NORM(x_p - x);

        //

        x = x_p;
    }

    //

    error_reporting(init_out_vals, x_p, opt_objfn, opt_data, success, err, err_tol, iter, iter_max, conv_failure_switch, settings_inp);
    
    return success;
}

optimlib_inline
bool
optim::newton(Vec_t& init_out_vals, 
              std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, Mat_t* hess_out, void* opt_data)> opt_objfn, 
              void* opt_data)
{
    return newton_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

optimlib_inline
bool
optim::newton(Vec_t& init_out_vals, 
              std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, Mat_t* hess_out, void* opt_data)> opt_objfn, 
              void* opt_data, 
              algo_settings_t& settings)
{
    return newton_int(init_out_vals,opt_objfn,opt_data,&settings);
}
