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
 * Generic input to constrained optimization routines
 *
 * Keith O'Hara
 * 01/11/2017
 *
 * This version:
 * 07/19/2017
 */

#include "optim.hpp"

bool optim::generic_constr_optim_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                                     std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
                                     double* value_out, optim_opt_settings* opt_params)
{
    return sumt_int(init_out_vals,opt_objfn,opt_data,constr_fn,constr_data,value_out,opt_params);
}

bool optim::generic_constr_optim(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                                 std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data)
{
    return generic_constr_optim_int(init_out_vals,opt_objfn,opt_data,constr_fn,constr_data,nullptr,nullptr);
}

bool optim::generic_constr_optim(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                                 std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
                                 optim_opt_settings& opt_params)
{
    return generic_constr_optim_int(init_out_vals,opt_objfn,opt_data,constr_fn,constr_data,nullptr,&opt_params);
}

bool optim::generic_constr_optim(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                                 std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
                                 double& value_out)
{
    return generic_constr_optim_int(init_out_vals,opt_objfn,opt_data,constr_fn,constr_data,&value_out,nullptr);
}

bool optim::generic_constr_optim(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                                 std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
                                 double& value_out, optim_opt_settings& opt_params)
{
    return generic_constr_optim_int(init_out_vals,opt_objfn,opt_data,constr_fn,constr_data,&value_out,&opt_params);
}

//
// box constraints

bool optim::generic_constr_optim_int(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
							         std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                                     std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
                                     double* value_out, optim_opt_settings* opt_params)
{
    // notation: 'p' stands for '+1'.
    //
    bool success = false;

    const int conv_failure_switch = (opt_params) ? opt_params->conv_failure_switch : OPTIM_CONV_FAILURE_POLICY;
    const int iter_max = (opt_params) ? opt_params->iter_max : OPTIM_DEFAULT_ITER_MAX;
    const double err_tol = (opt_params) ? opt_params->err_tol : OPTIM_DEFAULT_ERR_TOL;
    
    const double eta = (opt_params) ? opt_params->eta : OPTIM_DEFAULT_PENALTY_GROWTH; // growth of penalty parameter

    arma::vec x = init_out_vals;
    //
    // lambda function that combines the objective function with the constraints
    std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* sumt_data)> sumt_objfn = [opt_objfn, opt_data, constr_fn, constr_data] (const arma::vec& vals_inp, arma::vec* grad, void* sumt_data) -> double {
        sumt_struct *d = reinterpret_cast<sumt_struct*>(sumt_data);
        double c_pen = d->c_pen;
        //
        int n_vals = vals_inp.n_elem;
        arma::vec grad_obj(n_vals), grad_constr(n_vals);
        //
        //double ret = opt_objfn(vals_inp,grad_obj,opt_data) + c_pen*(std::pow(std::max(0.0,constr_fn(vals_inp,grad_constr,constr_data)),2) / 2.0);
        double ret = 1E06;
        double constr_val = constr_fn(vals_inp,&grad_constr,constr_data);

        if (constr_val < 0.0) {
            ret = opt_objfn(vals_inp,&grad_obj,opt_data);
            if (grad) {
                *grad = grad_obj;
            }
        } else {
            ret = opt_objfn(vals_inp,&grad_obj,opt_data) + c_pen*(constr_val*constr_val / 2.0);
            if (grad) {
                *grad = grad_obj + c_pen*grad_constr;
            }
        }
        //
        return ret;
    };
    //
    // initialization
    sumt_struct sumt_data;
    sumt_data.c_pen = 1.0;

    arma::vec x_p = x;
    //
    // begin loop
    int iter = 0;
    double err = 2*err_tol;

    while (err > err_tol && iter < iter_max) {
        iter++;
        //
        generic_optim_int(x_p,lower_bounds,upper_bounds,sumt_objfn,&sumt_data,nullptr,opt_params);
        err = arma::norm(x_p - x,2);
        //
        sumt_data.c_pen = eta*sumt_data.c_pen; // increase penalization parameter value
        x = x_p;
    }
    //
    error_reporting(init_out_vals,x_p,opt_objfn,opt_data,success,value_out,err,err_tol,iter,iter_max,conv_failure_switch);
    //
    return success;
}

bool optim::generic_constr_optim(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
						  std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data)
{
    return generic_constr_optim_int(init_out_vals,lower_bounds,upper_bounds,opt_objfn,opt_data,constr_fn,constr_data,nullptr,nullptr);
}

bool optim::generic_constr_optim(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
						  std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
                          optim_opt_settings& opt_params)
{
    return generic_constr_optim_int(init_out_vals,lower_bounds,upper_bounds,opt_objfn,opt_data,constr_fn,constr_data,nullptr,&opt_params);
}

bool optim::generic_constr_optim(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
						  std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
                          double& value_out)
{
    return generic_constr_optim_int(init_out_vals,lower_bounds,upper_bounds,opt_objfn,opt_data,constr_fn,constr_data,&value_out,nullptr);
}

bool optim::generic_constr_optim(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
						  std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
                          double& value_out, optim_opt_settings& opt_params)
{
    return generic_constr_optim_int(init_out_vals,lower_bounds,upper_bounds,opt_objfn,opt_data,constr_fn,constr_data,&value_out,&opt_params);
}
