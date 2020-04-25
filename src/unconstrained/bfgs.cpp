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
 * BFGS method for quasi-Newton-based non-linear optimization
 */

#include "optim.hpp"

// [OPTIM_BEGIN]
optimlib_inline
bool
optim::bfgs_int(Vec_t& init_out_vals, 
                std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
                void* opt_data, 
                algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.

    bool success = false;

    const size_t n_vals = OPTIM_MATOPS_SIZE(init_out_vals);

    //
    // BFGS settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }
    
    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const uint_t iter_max = settings.iter_max;
    const double err_tol = settings.err_tol;

    const double wolfe_cons_1 = 1E-03; // line search tuning parameters
    const double wolfe_cons_2 = 0.90;

    const bool vals_bound = settings.vals_bound;
    
    const Vec_t lower_bounds = settings.lower_bounds;
    const Vec_t upper_bounds = settings.upper_bounds;

    const VecInt_t bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    // lambda function for box constraints

    std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* box_data)> box_objfn \
    = [opt_objfn, vals_bound, bounds_type, lower_bounds, upper_bounds] (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data) \
    -> double
    {
        if (vals_bound) {
            Vec_t vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);
            double ret;
            
            if (grad_out) {
                Vec_t grad_obj = *grad_out;

                ret = opt_objfn(vals_inv_trans,&grad_obj,opt_data);

                // Mat_t jacob_matrix = jacobian_adjust(vals_inp,bounds_type,lower_bounds,upper_bounds);
                Vec_t jacob_vec = OPTIM_MATOPS_EXTRACT_DIAG( jacobian_adjust(vals_inp, bounds_type, lower_bounds, upper_bounds) );

                // *grad_out = jacob_matrix * grad_obj; // no need for transpose as jacob_matrix is diagonal
                *grad_out = OPTIM_MATOPS_HADAMARD_PROD(jacob_vec, grad_obj);
            } else {
                ret = opt_objfn(vals_inv_trans,nullptr,opt_data);
            }

            return ret;
        } else {
            return opt_objfn(vals_inp,grad_out,opt_data);
        }
    };

    //
    // initialization

    Vec_t x = init_out_vals;

    if (! OPTIM_MATOPS_IS_FINITE(x) ) {
        printf("bfgs error: non-finite initial value(s).\n");
        return false;
    }

    if (vals_bound) { // should we transform the parameters?
        x = transform(x, bounds_type, lower_bounds, upper_bounds);
    }

    const Mat_t I_mat = OPTIM_MATOPS_EYE(n_vals);
    Mat_t W = I_mat; // initial approx. to (inverse) Hessian 

    Vec_t grad(n_vals); // gradient vector
    box_objfn(x,&grad,opt_data);

    double err = OPTIM_MATOPS_L2NORM(grad);
    if (err <= err_tol) {
        return true;
    }

    //
    // if ||gradient(initial values)|| > tolerance, continue

    Vec_t d = - W*grad; // direction

    Vec_t x_p = x, grad_p = grad;

    line_search_mt(1.0, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, box_objfn, opt_data);

    err = OPTIM_MATOPS_L2NORM(grad);  // check updated values
    if (err <= err_tol) {
        init_out_vals = x_p;
        return true;
    }

    //
    // update W

    Vec_t s = x_p - x;
    Vec_t y = grad_p - grad;

    double W_denom_term = OPTIM_MATOPS_DOT_PROD(y,s);
    Mat_t W_term_1;

    if (W_denom_term > 1E-10) {   
        // checking the curvature condition y's > 0
        W_term_1 = I_mat - s * (OPTIM_MATOPS_TRANSPOSE_IN_PLACE(y)) / W_denom_term;
    
        W = W_term_1 * W * (OPTIM_MATOPS_TRANSPOSE_IN_PLACE(W_term_1)) + s * (OPTIM_MATOPS_TRANSPOSE_IN_PLACE(s)) / W_denom_term; // rank-1 update of inverse Hessian approximation
    } else {
        W = 0.1*W;
    }

    grad = grad_p;

    //
    // begin loop

    uint_t iter = 0;

    while (err > err_tol && iter < iter_max) {
        iter++;

        //

        d = - W*grad;
        line_search_mt(1.0, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, box_objfn, opt_data);
        
        err = OPTIM_MATOPS_L2NORM(grad_p);
        if (err <= err_tol) {
            break;
        }

        // if ||gradient(x_p)|| > tolerance, then continue

        s = x_p - x;
        y = grad_p - grad;

        W_denom_term = OPTIM_MATOPS_DOT_PROD(y,s);

        if (W_denom_term > 1E-10) {
            // checking the curvature condition y.s > 0
            W_term_1 = I_mat - s * OPTIM_MATOPS_TRANSPOSE_IN_PLACE(y) / W_denom_term;
        
            W = W_term_1 * W * OPTIM_MATOPS_TRANSPOSE_IN_PLACE(W_term_1) + s * OPTIM_MATOPS_TRANSPOSE_IN_PLACE(s) / W_denom_term;
        }

        //

        err = OPTIM_MATOPS_L2NORM(s);
        x = x_p;
        grad = grad_p;
    }

    //

    if (vals_bound) {
        x_p = inv_transform(x_p, bounds_type, lower_bounds, upper_bounds);
    }

    error_reporting(init_out_vals, x_p, opt_objfn, opt_data, success, err, err_tol, iter, iter_max, conv_failure_switch, settings_inp);

    //
    
    return success;
}

optimlib_inline
bool
optim::bfgs(Vec_t& init_out_vals, 
            std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
            void* opt_data)
{
    return bfgs_int(init_out_vals, opt_objfn, opt_data, nullptr);
}

optimlib_inline
bool
optim::bfgs(Vec_t& init_out_vals, 
            std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
            void* opt_data, 
            algo_settings_t& settings)
{
    return bfgs_int(init_out_vals, opt_objfn, opt_data, &settings);
}
