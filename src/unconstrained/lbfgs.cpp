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
 * L-BFGS method for quasi-Newton-based non-linear optimization
 */

#include "optim.hpp"

bool
optim::lbfgs_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.

    bool success = false;

    const size_t n_vals = init_out_vals.n_elem;

    //
    // L-BFGS settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }
    
    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const uint_t iter_max = settings.iter_max;
    const double err_tol = settings.err_tol;

    const double wolfe_cons_1 = 1E-03; // line search tuning parameters
    const double wolfe_cons_2 = 0.90;

    const uint_t par_M = settings.lbfgs_par_M; // how many previous iterations to use when updating the Hessian

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
        printf("lbfgs error: non-finite initial value(s).\n");
        return false;
    }

    if (vals_bound) { // should we transform the parameters?
        x = transform(x, bounds_type, lower_bounds, upper_bounds);
    }

    arma::vec grad(n_vals); // gradient vector
    box_objfn(x,&grad,opt_data);

    double err = arma::norm(grad, 2);
    if (err <= err_tol) {
        return true;
    }

    //
    // if ||gradient(initial values)|| > tolerance, then continue

    arma::vec d = - grad; // direction

    arma::vec x_p = x, grad_p = grad;

    line_search_mt(1.0, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, box_objfn, opt_data);

    err = arma::norm(grad, 2);  // check updated values
    if (err <= err_tol)
    {
        init_out_vals = x_p;
        return true;
    }

    //

    arma::vec s = x_p - x;
    arma::vec y = grad_p - grad;

    arma::mat s_mat(n_vals,par_M);
    arma::mat y_mat(n_vals,par_M);

    s_mat.col(0) = s;
    y_mat.col(0) = y;

    grad = grad_p;

    //
    // begin loop

    uint_t iter = 0;

    while (err > err_tol && iter < iter_max) 
    {
        iter++;

        //

        d = - lbfgs_recur(grad,s_mat,y_mat,std::min(iter,par_M));

        line_search_mt(1.0, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, box_objfn, opt_data);
        
        err = arma::norm(grad_p, 2);
        if (err <= err_tol) {
            break;
        }

        // if ||gradient(x_p)|| > tolerance, then continue

        s = x_p - x;
        y = grad_p - grad;

        err = arma::norm(s, 2);

        //

        x = x_p;
        grad = grad_p;

        s_mat.cols(1,par_M-1) = s_mat.cols(0,par_M-2);
        y_mat.cols(1,par_M-1) = y_mat.cols(0,par_M-2);

        s_mat.col(0) = s;
        y_mat.col(0) = y;
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
optim::lbfgs(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data)
{
    return lbfgs_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

bool
optim::lbfgs(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t& settings)
{
    return lbfgs_int(init_out_vals,opt_objfn,opt_data,&settings);
}

// algorithm 7.4 of Nocedal and Wright (2006)
arma::vec
optim::lbfgs_recur(arma::vec q, const arma::mat& s_mat, const arma::mat& y_mat, const uint_t M)
{
    arma::vec alpha_vec(M);

    // forwards

    double rho = 1.0;

    for (size_t i=0; i < M; i++) 
    {
        rho = 1.0 / arma::dot(y_mat.col(i),s_mat.col(i));
        alpha_vec(i) = rho*arma::dot(s_mat.col(i),q);

        q -= alpha_vec(i)*y_mat.col(i);
    }

    arma::vec r = q * ( arma::dot(s_mat.col(0),y_mat.col(0)) / arma::dot(y_mat.col(0),y_mat.col(0)) );

    // backwards

    double beta = 1.0;

    for (int i = M - 1; i >= 0; i--) 
    {
        rho = 1.0 / arma::dot(y_mat.col(i),s_mat.col(i));
        beta = rho*arma::dot(y_mat.col(i),r);

        r += (alpha_vec(i) - beta)*s_mat.col(i);
    }

    return r;
}
