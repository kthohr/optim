/*################################################################################
  ##
  ##   Copyright (C) 2016-2022 Keith O'Hara
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
optim::internal::bfgs_impl(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t* settings_inp
)
{
    // notation: 'p' stands for '+1'.

    bool success = false;

    const size_t n_vals = BMO_MATOPS_SIZE(init_out_vals);

    //
    // BFGS settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int print_level = settings.print_level;
    
    const uint_t conv_failure_switch = settings.conv_failure_switch;

    const size_t iter_max = settings.iter_max;
    const fp_t grad_err_tol = settings.grad_err_tol;
    const fp_t rel_sol_change_tol = settings.rel_sol_change_tol;

    const fp_t wolfe_cons_1 = settings.bfgs_settings.wolfe_cons_1; // line search tuning parameter
    const fp_t wolfe_cons_2 = settings.bfgs_settings.wolfe_cons_2; // line search tuning parameter

    const bool vals_bound = settings.vals_bound;
    
    const ColVec_t lower_bounds = settings.lower_bounds;
    const ColVec_t upper_bounds = settings.upper_bounds;

    const ColVecInt_t bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    // lambda function for box constraints

    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* box_data)> box_objfn \
    = [opt_objfn, vals_bound, bounds_type, lower_bounds, upper_bounds] (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data) \
    -> fp_t
    {
        if (vals_bound) {
            ColVec_t vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);
            fp_t ret;
            
            if (grad_out) {
                ColVec_t grad_obj = *grad_out;

                ret = opt_objfn(vals_inv_trans,&grad_obj,opt_data);

                // Mat_t jacob_matrix = jacobian_adjust(vals_inp,bounds_type,lower_bounds,upper_bounds);
                ColVec_t jacob_vec = BMO_MATOPS_EXTRACT_DIAG( jacobian_adjust(vals_inp, bounds_type, lower_bounds, upper_bounds) );

                // *grad_out = jacob_matrix * grad_obj; // no need for transpose as jacob_matrix is diagonal
                *grad_out = BMO_MATOPS_HADAMARD_PROD(jacob_vec, grad_obj);
            } else {
                ret = opt_objfn(vals_inv_trans,nullptr,opt_data);
            }

            return ret;
        } else {
            return opt_objfn(vals_inp,grad_out,opt_data);
        }
    };

    // initialization

    ColVec_t x = init_out_vals;

    if (! BMO_MATOPS_IS_FINITE(x) ) {
        printf("bfgs error: non-finite initial value(s).\n");
        return false;
    }

    if (vals_bound) { // should we transform the parameters?
        x = transform(x, bounds_type, lower_bounds, upper_bounds);
    }

    const Mat_t I_mat = BMO_MATOPS_EYE(n_vals);

    Mat_t W = I_mat;                                // initial approx. to (inverse) Hessian 
    ColVec_t grad(n_vals);                          // gradient vector
    ColVec_t d = BMO_MATOPS_ZERO_COLVEC(n_vals);    // direction vector
    ColVec_t s = BMO_MATOPS_ZERO_COLVEC(n_vals);
    ColVec_t y = BMO_MATOPS_ZERO_COLVEC(n_vals);

    box_objfn(x, &grad, opt_data);

    fp_t grad_err = BMO_MATOPS_L2NORM(grad);

    OPTIM_BFGS_TRACE(-1, grad_err, 0.0, x, d, grad, s, y, W);

    if (grad_err <= grad_err_tol) {
        return true;
    }

    // if ||gradient(initial values)|| > tolerance, continue

    d = - W*grad; // direction

    ColVec_t x_p = x, grad_p = grad;

    line_search_mt(1.0, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, box_objfn, opt_data);

    s = x_p - x;
    y = grad_p - grad;

    // update approx. inverse Hessian (W)

    fp_t W_denom_term = BMO_MATOPS_DOT_PROD(y,s);
    Mat_t W_term_1;

    if (W_denom_term > 1E-10) {   
        // checking whether the curvature condition holds: y's > 0
        W_term_1 = I_mat - s * (BMO_MATOPS_TRANSPOSE_INPLACE(y)) / W_denom_term;
    
        // perform rank-1 update of inverse Hessian approximation
        W = W_term_1 * W * (BMO_MATOPS_TRANSPOSE_INPLACE(W_term_1)) + s * (BMO_MATOPS_TRANSPOSE_INPLACE(s)) / W_denom_term;
    } else {
        W = 0.1 * W;
    }

    grad = grad_p;

    grad_err = BMO_MATOPS_L2NORM(grad_p);
    fp_t rel_sol_change = BMO_MATOPS_L1NORM( BMO_MATOPS_ARRAY_DIV_ARRAY(s, (BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ABS(x), OPTIM_FPN_SMALL_NUMBER)) ) );

    OPTIM_BFGS_TRACE(0, grad_err, rel_sol_change, x_p, d, grad_p, s, y, W);

    if (grad_err <= grad_err_tol) {
    	if (vals_bound) {
    	    init_out_vals = inv_transform(x_p, bounds_type, lower_bounds, upper_bounds);
    	} else {
            init_out_vals = x_p;
        }
        return true;
    }

    // begin loop

    size_t iter = 0;

    while (grad_err > grad_err_tol && rel_sol_change > rel_sol_change_tol && iter < iter_max) {
        ++iter;

        //

        d = - W*grad;

        line_search_mt(1.0, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, box_objfn, opt_data);

        //

        s = x_p - x;
        y = grad_p - grad;

        W_denom_term = BMO_MATOPS_DOT_PROD(y,s);

        if (W_denom_term > 1E-10) {
            // checking the curvature condition y.s > 0
            W_term_1 = I_mat - s * BMO_MATOPS_TRANSPOSE_INPLACE(y) / W_denom_term;
        
            W = W_term_1 * W * BMO_MATOPS_TRANSPOSE_INPLACE(W_term_1) + s * BMO_MATOPS_TRANSPOSE_INPLACE(s) / W_denom_term;
        }

        //

        grad_err = BMO_MATOPS_L2NORM(grad_p);
        rel_sol_change = BMO_MATOPS_L1NORM( BMO_MATOPS_ARRAY_DIV_ARRAY(s, (BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ABS(x), OPTIM_FPN_SMALL_NUMBER)) ) );
        
        x = x_p;
        grad = grad_p;

        //
    
        OPTIM_BFGS_TRACE(iter, grad_err, rel_sol_change, x, d, grad, s, y, W);
    }

    //

    if (vals_bound) {
        x_p = inv_transform(x_p, bounds_type, lower_bounds, upper_bounds);
    }

    error_reporting(init_out_vals, x_p, opt_objfn, opt_data, 
                    success, grad_err, grad_err_tol, iter, iter_max, 
                    conv_failure_switch, settings_inp);

    //
    
    return success;
}

optimlib_inline
bool
optim::bfgs(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data
)
{
    return internal::bfgs_impl(init_out_vals, opt_objfn, opt_data, nullptr);
}

optimlib_inline
bool
optim::bfgs(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t& settings
)
{
    return internal::bfgs_impl(init_out_vals, opt_objfn, opt_data, &settings);
}
