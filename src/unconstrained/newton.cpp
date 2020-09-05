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
optim::internal::newton_impl(
    Vec_t& init_out_vals, 
    std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, Mat_t* hess_out, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.

    bool success = false;

    const size_t n_vals = OPTIM_MATOPS_SIZE(init_out_vals);

    // settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int print_level = settings.print_level;
    
    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const size_t iter_max = settings.iter_max;
    const double grad_err_tol = settings.grad_err_tol;
    const double rel_sol_change_tol = settings.rel_sol_change_tol;

    // initialization

    Vec_t x = init_out_vals;
    Vec_t x_p = x;

    if (! OPTIM_MATOPS_IS_FINITE(x) ) {
        printf("newton error: non-finite initial value(s).\n");
        return false;
    }

    Mat_t H(n_vals, n_vals);                    // hessian matrix
    Vec_t grad(n_vals);                         // gradient vector
    Vec_t d = OPTIM_MATOPS_ZERO_VEC(n_vals);    // direction vector

    opt_objfn(x_p, &grad, &H, opt_data);

    double grad_err = OPTIM_MATOPS_L2NORM(grad);

    OPTIM_NEWTON_TRACE(-1, grad_err, 0.0, x_p, d, grad, H);

    if (grad_err <= grad_err_tol) {
        return true;
    }

    // if ||gradient(initial values)|| > tolerance, then continue

    d = - OPTIM_MATOPS_SOLVE(H, grad); // Newton direction

    x_p += d; // no line search used here

    opt_objfn(x_p, &grad, &H, opt_data);

    grad_err = OPTIM_MATOPS_L2NORM(grad);
    double rel_sol_change = OPTIM_MATOPS_L1NORM( OPTIM_MATOPS_ARRAY_DIV_ARRAY( (x_p - x), (OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_ABS(x), 1.0e-08)) ) );

    OPTIM_NEWTON_TRACE(0, grad_err, rel_sol_change, x_p, d, grad, H);

    if (grad_err <= grad_err_tol) {
        init_out_vals = x_p;
        return true;
    }

    x = x_p;

    // begin loop

    size_t iter = 0;

    while (grad_err > grad_err_tol && rel_sol_change > rel_sol_change_tol && iter < iter_max) {
        ++iter;

        //

        d = - OPTIM_MATOPS_SOLVE(H,grad);
        x_p += d;
        
        opt_objfn(x_p, &grad, &H, opt_data);
        
        //

        grad_err = OPTIM_MATOPS_L2NORM(grad);
        rel_sol_change = OPTIM_MATOPS_L1NORM( OPTIM_MATOPS_ARRAY_DIV_ARRAY( (x_p - x), (OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_ABS(x), 1.0e-08)) ) );

        //

        x = x_p;
    
        OPTIM_NEWTON_TRACE(iter, grad_err, rel_sol_change, x_p, d, grad, H);
    }

    //

    error_reporting(init_out_vals, x_p, opt_objfn, opt_data, 
                    success, grad_err, grad_err_tol, iter, iter_max, 
                    conv_failure_switch, settings_inp);
    
    return success;
}

optimlib_inline
bool
optim::newton(Vec_t& init_out_vals, 
              std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, Mat_t* hess_out, void* opt_data)> opt_objfn, 
              void* opt_data)
{
    return internal::newton_impl(init_out_vals,opt_objfn,opt_data,nullptr);
}

optimlib_inline
bool
optim::newton(Vec_t& init_out_vals, 
              std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, Mat_t* hess_out, void* opt_data)> opt_objfn, 
              void* opt_data, 
              algo_settings_t& settings)
{
    return internal::newton_impl(init_out_vals,opt_objfn,opt_data,&settings);
}
