/*################################################################################
  ##
  ##   Copyright (C) 2016-2017 Keith O'Hara
  ##
  ##   This file is part of the OptimLib C++ library.
  ##
  ##   OptimLib is free software: you can redistribute it and/or modify
  ##   it under the terms of the GNU General Public License as published by
  ##   the Free Software Foundation, either version 2 of the License, or
  ##   (at your option) any later version.
  ##
  ##   OptimLib is distributed in the hope that it will be useful,
  ##   but WITHOUT ANY WARRANTY; without even the implied warranty of
  ##   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  ##   GNU General Public License for more details.
  ##
  ################################################################################*/

/*
 * BFGS method for quasi-Newton-based non-linear optimization
 *
 * Keith O'Hara
 * 12/23/2016
 *
 * This version:
 * 07/18/2017
 */

#include "optim.hpp"

bool
optim::bfgs_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data, double* value_out, optim_opt_settings* opt_params)
{
    // notation: 'p' stands for '+1'.
    //
    bool success = false;
    
    const int conv_failure_switch = (opt_params) ? opt_params->conv_failure_switch : OPTIM_CONV_FAILURE_POLICY;
    const int iter_max = (opt_params) ? opt_params->iter_max : OPTIM_DEFAULT_ITER_MAX;
    const double err_tol = (opt_params) ? opt_params->err_tol : OPTIM_DEFAULT_ERR_TOL;

    const double wolfe_cons_1 = 1E-03; // line search tuning parameters
    const double wolfe_cons_2 = 0.90;
    //
    int n_vals = init_out_vals.n_elem;
    arma::vec x = init_out_vals;

    arma::mat W = arma::eye(n_vals,n_vals); // initial approx. to (inverse) Hessian 
    arma::mat I_mat = arma::eye(n_vals,n_vals);
    //
    // initialization
    arma::vec grad(n_vals); // gradient
    opt_objfn(x,&grad,opt_data);

    double err = arma::accu(arma::abs(grad));
    if (err <= err_tol) {
        return true;
    }
    // if ||gradient(initial values)|| > tolerance, then continue
    double t_line = 1.0; // initial line search value
    arma::vec d = - W*grad; // direction

    arma::vec x_p = x, grad_p = grad;
    // line_search_mt(t_line, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, opt_objfn, opt_data);
    t_line = line_search_mt(1.0, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, opt_objfn, opt_data);

    err = arma::accu(arma::abs(grad_p)); // check updated values
    if (err <= err_tol) {
        init_out_vals = x_p;
        return true;
    }

    if (t_line < 1E-14) {
        // x_p.ones();
        x_p.randu();
        t_line = line_search_mt(1.0, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, opt_objfn, opt_data);
    }
    // if ||gradient(x_p)|| > tolerance, then continue
    arma::vec s = x_p - x;
    arma::vec y = grad_p - grad;

    if (arma::norm(s,2) < 1E-14) {
        init_out_vals = x_p;
        return true;
    }

    // update W
    double W_denom_term = arma::dot(y,s);

    arma::mat W_term_1 = I_mat - s*y.t() / W_denom_term;
    W = W_term_1*W*W_term_1.t() + s*s.t() / W_denom_term; // update inverse Hessian approximation

    grad = grad_p;
    //
    // begin loop
    int iter = 0;

    while (err > err_tol && iter < iter_max) {
        iter++;
        //
        d = - W*grad;
        t_line = line_search_mt(1.0, x_p, grad_p, d, &wolfe_cons_1, &wolfe_cons_2, opt_objfn, opt_data);
        
        // err = arma::accu(arma::abs(grad_p));
        err = arma::norm(grad_p, 2);
        if (err <= err_tol) {
            break;
        }

        // if ||gradient(x_p)|| > tolerance, then continue
        s = x_p - x;
        y = grad_p - grad;

        err = arma::norm(s, 2);

        // update W
        W_denom_term = arma::dot(y,s);

        if (W_denom_term > 1E-12) { // checking the curvature condition
            W_term_1 = I_mat - s*y.t() / W_denom_term;
        
            W = W_term_1*W*W_term_1.t() + s*s.t() / W_denom_term;
        }
        //
        x = x_p;
        grad = grad_p;
    }
    //
    error_reporting(init_out_vals,x_p,opt_objfn,opt_data,success,value_out,err,err_tol,iter,iter_max,conv_failure_switch);
    //
    return success;
}

bool
optim::bfgs(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data)
{
    return bfgs_int(init_out_vals,opt_objfn,opt_data,NULL,NULL);
}

bool
optim::bfgs(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data, optim_opt_settings& opt_params)
{
    return bfgs_int(init_out_vals,opt_objfn,opt_data,NULL,&opt_params);
}

bool
optim::bfgs(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data, double& value_out)
{
    return bfgs_int(init_out_vals,opt_objfn,opt_data,&value_out,NULL);
}

bool
optim::bfgs(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data, double& value_out, optim_opt_settings& opt_params)
{
    return bfgs_int(init_out_vals,opt_objfn,opt_data,&value_out,&opt_params);
}
