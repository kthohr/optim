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

/*
 * Conjugate Gradient (CG) method for non-linear optimization
 */

#include "optim.hpp"

bool
optim::cg_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.
    //
    bool success = false;
    
    const size_t n_vals = init_out_vals.n_elem;

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
    
    const arma::vec lower_bounds = settings.lower_bounds;
    const arma::vec upper_bounds = settings.upper_bounds;

    const arma::uvec bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    // lambda function for box constraints

    std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* box_data)> box_objfn \
    = [opt_objfn, vals_bound, bounds_type, lower_bounds, upper_bounds] (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data) \
    -> double 
    {
        if (vals_bound)
        {
            arma::vec vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);
            double ret;
            
            if (grad_out)
            {
                arma::vec grad_obj = *grad_out;

                ret = opt_objfn(vals_inv_trans,&grad_obj,opt_data);

                // arma::mat jacob_matrix = jacobian_adjust(vals_inp,bounds_type,lower_bounds,upper_bounds);
                arma::vec jacob_vec = arma::diagvec(jacobian_adjust(vals_inp,bounds_type,lower_bounds,upper_bounds));

                // *grad_out = jacob_matrix * grad_obj; // no need for transpose as jacob_matrix is diagonal
                *grad_out = jacob_vec % grad_obj;
            }
            else
            {
                ret = opt_objfn(vals_inv_trans,nullptr,opt_data);
            }

            return ret;
        }
        else
        {
            return opt_objfn(vals_inp,grad_out,opt_data);
        }
    };

    //
    // initialization

    arma::vec x = init_out_vals;

    if (!x.is_finite())
    {
        printf("cg error: non-finite initial value(s).\n");
        return false;
    }

    if (vals_bound) { // should we transform the parameters?
        x = transform(x, bounds_type, lower_bounds, upper_bounds);
    }

    arma::vec grad(n_vals); // gradient
    box_objfn(x,&grad,opt_data);

    // double err = arma::accu(arma::abs(grad));
    double err = arma::norm(grad, 2);
    if (err <= err_tol) {
        return true;
    }

    //

    double t_init = 1.0; // initial value for line search

    arma::vec d = - grad, d_p;
    arma::vec x_p = x, grad_p = grad;

    double t = line_search_mt(t_init, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, box_objfn, opt_data);

    err = arma::norm(grad_p, 2);
    if (err <= err_tol)
    {
        init_out_vals = x_p;
        return true;
    }

    //
    // begin loop

    uint_t iter = 0;

    while (err > err_tol && iter < iter_max)
    {
        iter++;

        //

        double beta = cg_update(grad,grad_p,d,iter,cg_method,cg_restart_threshold);
        d_p = - grad_p + beta*d;

        t_init = t * (arma::dot(grad,d) / arma::dot(grad_p,d_p));

        grad = grad_p;

        t = line_search_mt(t_init, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, box_objfn, opt_data);

        //

        err = arma::norm(grad_p, 2);
        d = d_p;
        x = x_p;
    }

    //

    if (vals_bound) {
        x_p = inv_transform(x_p, bounds_type, lower_bounds, upper_bounds);
    }

    error_reporting(init_out_vals,x_p,opt_objfn,opt_data,success,err,err_tol,iter,iter_max,conv_failure_switch,settings_inp);

    //

    return success;
}

bool
optim::cg(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data)
{
    return cg_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

bool
optim::cg(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t& settings)
{
    return cg_int(init_out_vals,opt_objfn,opt_data,&settings);
}

//
// update formula

double optim::cg_update(const arma::vec& grad, const arma::vec& grad_p, const arma::vec& direc, const uint_t iter, const uint_t cg_method, const double cg_restart_threshold)
{
    // threshold test
    double ratio_value = std::abs( arma::dot(grad_p,grad) ) / arma::dot(grad_p,grad_p);

    if ( ratio_value > cg_restart_threshold )
    {
        return 0.0;
    }
    else
    {
        double beta = 1.0;

        switch (cg_method)
        {
            case 1: // Fletcher-Reeves (FR)
            {
                beta = arma::dot(grad_p,grad_p) / arma::dot(grad,grad);
                break;
            }

            case 2: // Polak-Ribiere (PR) + 
            {
                beta = arma::dot(grad_p,grad_p - grad) / arma::dot(grad,grad); // max(.,0.0) moved to end
                break;
            }

            case 3: // FR-PR hybrid, see eq. 5.48 in Nocedal and Wright
            {
                if (iter > 1) 
                {
                    const double beta_FR = arma::dot(grad_p,grad_p) / arma::dot(grad,grad);
                    const double beta_PR = arma::dot(grad_p,grad_p - grad) / arma::dot(grad,grad);
                    
                    if (beta_PR < - beta_FR) {
                        beta = -beta_FR;
                    } else if (std::abs(beta_PR) <= beta_FR) {
                        beta = beta_PR;
                    } else { // beta_PR > beta_FR
                        beta = beta_FR;
                    }
                } 
                else 
                {   // default to PR+
                    beta = arma::dot(grad_p,grad_p - grad) / arma::dot(grad,grad); // max(.,0.0) moved to end
                }
                break;
            }

            case 4: // Hestenes-Stiefel
            {
                beta = arma::dot(grad_p,grad_p - grad) / arma::dot(grad_p - grad,direc);
                break;
            }

            case 5: // Dai-Yuan
            {
                beta = arma::dot(grad_p,grad_p) / arma::dot(grad_p - grad,direc);
                break;
            }

            case 6: // Hager-Zhang
            {
                arma::vec y = grad_p - grad;

                arma::vec term_1 = y - 2*direc*(arma::dot(y,y) / arma::dot(y,direc));
                arma::vec term_2 = grad_p / arma::dot(y,direc);

                beta = arma::dot(term_1,term_2);
                break;
            }
            
            default:
            {
                printf("error: unknown value for cg_method");
                break;
            }
        }

        //

        return std::max(beta,0.0);
    }
}
