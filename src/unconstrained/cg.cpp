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
optim::cg_int(Vec_t& init_out_vals, 
              std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
              void* opt_data, 
              algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.
    
    bool success = false;
    
    const size_t n_vals = OPTIM_MATOPS_SIZE(init_out_vals);

    //
    // CG settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }
    
    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const uint_t iter_max = settings.iter_max;
    const double err_tol = settings.err_tol;

    const uint_t cg_method = settings.cg_method; // update method
    const double cg_restart_threshold = settings.cg_restart_threshold;

    const double wolfe_cons_1 = 1E-03; // line search tuning parameters
    const double wolfe_cons_2 = 0.10;

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

    //
    // initialization

    Vec_t x = init_out_vals;

    if (! OPTIM_MATOPS_IS_FINITE(x) ) {
        printf("cg error: non-finite initial value(s).\n");
        return false;
    }

    if (vals_bound) { // should we transform the parameters?
        x = transform(x, bounds_type, lower_bounds, upper_bounds);
    }

    Vec_t grad(n_vals); // gradient
    box_objfn(x, &grad, opt_data);

    // double err = arma::accu(arma::abs(grad));
    double err = OPTIM_MATOPS_L2NORM(grad);
    if (err <= err_tol) {
        return true;
    }

    //

    double t_init = 1.0; // initial value for line search

    Vec_t d = - grad, d_p;
    Vec_t x_p = x, grad_p = grad;

    double t = line_search_mt(t_init, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, box_objfn, opt_data);

    err = OPTIM_MATOPS_L2NORM(grad_p);
    if (err <= err_tol) {
        init_out_vals = x_p;
        return true;
    }

    //
    // begin loop

    uint_t iter = 0;

    while (err > err_tol && iter < iter_max) {
        ++iter;

        //

        double beta = cg_update(grad,grad_p,d,iter,cg_method,cg_restart_threshold);
        d_p = - grad_p + beta*d;

        t_init = t * (OPTIM_MATOPS_DOT_PROD(grad,d) / OPTIM_MATOPS_DOT_PROD(grad_p,d_p));

        grad = grad_p;

        t = line_search_mt(t_init, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, box_objfn, opt_data);

        //

        err = OPTIM_MATOPS_L2NORM(grad_p);

        d = d_p;
        x = x_p;
    }

    //

    if (vals_bound) {
        x_p = inv_transform(x_p, bounds_type, lower_bounds, upper_bounds);
    }

    error_reporting(init_out_vals, x_p,opt_objfn, opt_data, success, err, err_tol, iter, iter_max, conv_failure_switch, settings_inp);

    //

    return success;
}

optimlib_inline
bool
optim::cg(Vec_t& init_out_vals, 
          std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
          void* opt_data)
{
    return cg_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

optimlib_inline
bool
optim::cg(Vec_t& init_out_vals, 
          std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
          void* opt_data, 
          algo_settings_t& settings)
{
    return cg_int(init_out_vals,opt_objfn,opt_data,&settings);
}

//
// update formula

optimlib_inline
double
optim::cg_update(const Vec_t& grad, 
                 const Vec_t& grad_p, 
                 const Vec_t& direc, 
                 const uint_t iter, 
                 const uint_t cg_method, 
                 const double cg_restart_threshold)
{
    // threshold test
    double ratio_value = std::abs( OPTIM_MATOPS_DOT_PROD(grad_p,grad) ) / OPTIM_MATOPS_DOT_PROD(grad_p,grad_p);

    if ( ratio_value > cg_restart_threshold ) {
        return 0.0;
    } else {
        double beta = 1.0;

        switch (cg_method)
        {
            case 1: // Fletcher-Reeves (FR)
            {
                beta = OPTIM_MATOPS_DOT_PROD(grad_p,grad_p) / OPTIM_MATOPS_DOT_PROD(grad,grad);
                break;
            }

            case 2: // Polak-Ribiere (PR) + 
            {
                beta = OPTIM_MATOPS_DOT_PROD(grad_p, grad_p - grad) / OPTIM_MATOPS_DOT_PROD(grad,grad); // max(.,0.0) moved to end
                break;
            }

            case 3: // FR-PR hybrid, see eq. 5.48 in Nocedal and Wright
            {
                if (iter > 1) {
                    const double beta_denom = OPTIM_MATOPS_DOT_PROD(grad, grad);
                    
                    const double beta_FR = OPTIM_MATOPS_DOT_PROD(grad_p, grad_p) / beta_denom;
                    const double beta_PR = OPTIM_MATOPS_DOT_PROD(grad_p, grad_p - grad) / beta_denom;
                    
                    if (beta_PR < - beta_FR) {
                        beta = -beta_FR;
                    } else if (std::abs(beta_PR) <= beta_FR) {
                        beta = beta_PR;
                    } else { // beta_PR > beta_FR
                        beta = beta_FR;
                    }
                } else {
                    // default to PR+
                    beta = OPTIM_MATOPS_DOT_PROD(grad_p,grad_p - grad) / OPTIM_MATOPS_DOT_PROD(grad,grad); // max(.,0.0) moved to end
                }
                break;
            }

            case 4: // Hestenes-Stiefel
            {
                beta = OPTIM_MATOPS_DOT_PROD(grad_p,grad_p - grad) / OPTIM_MATOPS_DOT_PROD(grad_p - grad,direc);
                break;
            }

            case 5: // Dai-Yuan
            {
                beta = OPTIM_MATOPS_DOT_PROD(grad_p,grad_p) / OPTIM_MATOPS_DOT_PROD(grad_p - grad,direc);
                break;
            }

            case 6: // Hager-Zhang
            {
                Vec_t y = grad_p - grad;

                Vec_t term_1 = y - 2*direc*(OPTIM_MATOPS_DOT_PROD(y,y) / OPTIM_MATOPS_DOT_PROD(y,direc));
                Vec_t term_2 = grad_p / OPTIM_MATOPS_DOT_PROD(y,direc);

                beta = OPTIM_MATOPS_DOT_PROD(term_1,term_2);
                break;
            }
            
            default:
            {
                printf("error: unknown value for cg_method");
                break;
            }
        }

        //

        return std::max(beta, 0.0);
    }
}
