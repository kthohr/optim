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
 * Conjugate Gradient (CG) method for non-linear optimization
 */

#include "optim.hpp"

// [OPTIM_BEGIN]
optimlib_inline
bool
optim::internal::cg_impl(
    Vec_t& init_out_vals, 
    std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.
    
    bool success = false;
    
    const size_t n_vals = OPTIM_MATOPS_SIZE(init_out_vals);

    // CG settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int print_level = settings.print_level;
    
    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const size_t iter_max = settings.iter_max;
    const double grad_err_tol = settings.grad_err_tol;
    double rel_sol_change_tol = settings.rel_sol_change_tol;

    if (!settings.cg_settings.use_rel_sol_change_crit) {
        rel_sol_change_tol = -1.0;
    }

    const uint_t cg_method = settings.cg_settings.method; // cg update method
    const double cg_restart_threshold = settings.cg_settings.restart_threshold;

    const double wolfe_cons_1 = settings.cg_settings.wolfe_cons_1; // line search tuning parameter
    const double wolfe_cons_2 = settings.cg_settings.wolfe_cons_2;

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
                Vec_t jacob_vec = OPTIM_MATOPS_EXTRACT_DIAG( jacobian_adjust(vals_inp,bounds_type,lower_bounds,upper_bounds) );

                // *grad_out = jacob_matrix * grad_obj; // no need for transpose as jacob_matrix is diagonal
                *grad_out = OPTIM_MATOPS_HADAMARD_PROD(jacob_vec, grad_obj);
            } else {
                ret = opt_objfn(vals_inv_trans, nullptr, opt_data);
            }

            return ret;
        } else {
            return opt_objfn(vals_inp, grad_out, opt_data);
        }
    };

    // initialization

    if (! OPTIM_MATOPS_IS_FINITE(init_out_vals) ) {
        printf("gd error: non-finite initial value(s).\n");
        return false;
    }

    Vec_t x = init_out_vals;
    Vec_t d = OPTIM_MATOPS_ZERO_VEC(n_vals);

    if (vals_bound) { // should we transform the parameters?
        x = transform(x, bounds_type, lower_bounds, upper_bounds);
    }

    Vec_t grad(n_vals); // gradient
    box_objfn(x, &grad, opt_data);

    double grad_err = OPTIM_MATOPS_L2NORM(grad);

    OPTIM_CG_TRACE(-1, grad_err, 0.0, x, d, grad, 0.0);

    if (grad_err <= grad_err_tol) {
        return true;
    }

    //

    double t_init = 1.0; // initial value for line search

    d = - grad;
    Vec_t x_p = x, grad_p = grad;

    double t = line_search_mt(t_init, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, box_objfn, opt_data);

    grad_err = OPTIM_MATOPS_L2NORM(grad_p);
    double rel_sol_change = OPTIM_MATOPS_L1NORM( OPTIM_MATOPS_ARRAY_DIV_ARRAY( (x_p - x), (OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_ABS(x), 1.0e-08)) ) );
    
    OPTIM_CG_TRACE(0, grad_err, rel_sol_change, x, d, grad, 0.0);

    if (grad_err <= grad_err_tol) {
        init_out_vals = x_p;
        return true;
    }

    // begin loop

    size_t iter = 0;

    while (grad_err > grad_err_tol && iter < iter_max && rel_sol_change > rel_sol_change_tol) {
        ++iter;

        //

        double beta = cg_update(grad, grad_p, d, iter, cg_method, cg_restart_threshold);

        Vec_t d_p = - grad_p + beta*d;

        t_init = t * (OPTIM_MATOPS_DOT_PROD(grad,d) / OPTIM_MATOPS_DOT_PROD(grad_p,d_p));

        grad = grad_p;

        t = line_search_mt(t_init, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, box_objfn, opt_data);

        //

        grad_err = OPTIM_MATOPS_L2NORM(grad_p);
        rel_sol_change = OPTIM_MATOPS_L1NORM( OPTIM_MATOPS_ARRAY_DIV_ARRAY( (x_p - x), (OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_ABS(x), 1.0e-08)) ) );

        d = d_p;
        x = x_p;

        //

        OPTIM_CG_TRACE(iter, grad_err, rel_sol_change, x, d, grad, beta);
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
optim::cg(
    Vec_t& init_out_vals, 
    std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data)
{
    return internal::cg_impl(init_out_vals,opt_objfn,opt_data,nullptr);
}

optimlib_inline
bool
optim::cg(
    Vec_t& init_out_vals, 
    std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t& settings)
{
    return internal::cg_impl(init_out_vals,opt_objfn,opt_data,&settings);
}
