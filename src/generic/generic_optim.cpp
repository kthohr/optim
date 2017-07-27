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
 * Generic input to optimization routines
 *
 * Keith O'Hara
 * 01/11/2017
 *
 * This version:
 * 07/19/2017
 */

#include "optim.hpp"

bool optim::generic_optim_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                              double* value_out, optim_opt_settings* opt_params)
{
    return bfgs_int(init_out_vals,opt_objfn,opt_data,value_out,opt_params);
}

bool optim::generic_optim(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data)
{
    return generic_optim_int(init_out_vals,opt_objfn,opt_data,nullptr,nullptr);
}

bool optim::generic_optim(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data, 
                          optim_opt_settings& opt_params)
{
    return generic_optim_int(init_out_vals,opt_objfn,opt_data,nullptr,&opt_params);
}

bool optim::generic_optim(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data, 
                          double& value_out)
{
    return generic_optim_int(init_out_vals,opt_objfn,opt_data,&value_out,nullptr);
}

bool optim::generic_optim(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data, 
                          double& value_out, optim_opt_settings& opt_params)
{
    return generic_optim_int(init_out_vals,opt_objfn,opt_data,&value_out,&opt_params);
}

//
// box constraints

bool optim::generic_optim_int(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
                              std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                              double* value_out, optim_opt_settings* opt_params)
{
    const int conv_failure_switch = (opt_params) ? opt_params->conv_failure_switch : OPTIM_CONV_FAILURE_POLICY;
    //
    std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* box_data)> box_objfn = [opt_objfn, lower_bounds, upper_bounds] (const arma::vec& vals_inp, arma::vec* grad, void* opt_data) -> double {
        //
        arma::vec vals_inv_trans = logit_inv_trans(vals_inp,lower_bounds,upper_bounds);
        //
        double ret;
        
        if (grad) {
            arma::vec grad_obj = *grad;
            ret = opt_objfn(vals_inv_trans,&grad_obj,opt_data);

            arma::mat jacob_correct = jacob_matrix_logit(vals_inp,lower_bounds,upper_bounds);

            *grad = jacob_correct.t() * grad_obj; // correct gradient for transformation
        } else {
            ret = opt_objfn(vals_inv_trans,nullptr,opt_data);
        }
        //
        return ret;
    };
    //
    arma::vec initial_vals = logit_trans(init_out_vals,lower_bounds,upper_bounds);
    
    bool success = bfgs_int(initial_vals,box_objfn,opt_data,nullptr,opt_params);
    //
    initial_vals = logit_inv_trans(initial_vals,lower_bounds,upper_bounds);

    error_reporting(init_out_vals,initial_vals,opt_objfn,opt_data,success,value_out,conv_failure_switch);
    //
    return success;
}

bool optim::generic_optim(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
                          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data)
{
    return generic_optim_int(init_out_vals,lower_bounds,upper_bounds,opt_objfn,opt_data,nullptr,nullptr);
}

bool optim::generic_optim(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
                          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                          optim_opt_settings& opt_params)
{
    return generic_optim_int(init_out_vals,lower_bounds,upper_bounds,opt_objfn,opt_data,nullptr,&opt_params);
}

bool optim::generic_optim(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
                          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                          double& value_out)
{
    return generic_optim_int(init_out_vals,lower_bounds,upper_bounds,opt_objfn,opt_data,&value_out,nullptr);
}

bool optim::generic_optim(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
                          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                          double& value_out, optim_opt_settings& opt_params)
{
    return generic_optim_int(init_out_vals,lower_bounds,upper_bounds,opt_objfn,opt_data,&value_out,&opt_params);
}

//
// jacobian adjustment for box constraints

arma::mat optim::jacob_matrix_logit(const arma::vec& trans_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds)
{
    int n_vals = trans_vals.n_elem;
    arma::mat ret_mat = arma::zeros(n_vals,n_vals);

    for (int i=0; i < n_vals; i++) {
        ret_mat(i,i) = std::exp(trans_vals(i))*(upper_bounds(i) - lower_bounds(i)) / std::pow(std::exp(trans_vals(i)) + 1,2);
    }
    //
    return ret_mat;
}
