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
 * Sequential unconstrained minimization technique (SUMT)
 *
 * Keith O'Hara
 * 01/15/2016
 *
 * This version:
 * 07/31/2017
 */

#include "optim.hpp"

bool
optim::sumt_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data,
                std::function<arma::vec (const arma::vec& vals_inp, arma::mat* jacob_out, void* constr_data)> constr_fn, void* constr_data,
                double* value_out, opt_settings* settings_inp)
{
    // notation: 'p' stands for '+1'.
    //
    bool success = false;

    // const int n_vals = init_out_vals.n_elem;

    //
    // SUMT settings

    opt_settings settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int conv_failure_switch = settings.conv_failure_switch;
    const int iter_max = settings.iter_max;
    const double err_tol = settings.err_tol;

    const double par_eta = settings.sumt_par_eta; // growth of penalty parameter

    // lambda function that combines the objective function with the constraints

    std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* sumt_data)> sumt_objfn = [opt_objfn, opt_data, constr_fn, constr_data] (const arma::vec& vals_inp, arma::vec* grad_out, void* sumt_data) -> double {
        sumt_struct *d = reinterpret_cast<sumt_struct*>(sumt_data);
        double c_pen = d->c_pen;
        
        int n_vals = vals_inp.n_elem;
        arma::vec grad_obj(n_vals);
        arma::mat jacob_constr;
        //
        double ret = 1E08;

        arma::vec constr_vals = constr_fn(vals_inp,&jacob_constr,constr_data);
        arma::uvec z_inds = arma::find(constr_vals <= 0);

        if (z_inds.n_elem > 0) {
            constr_vals.elem(z_inds).zeros();
            jacob_constr.rows(z_inds).zeros();
        }

        double constr_valsq = arma::dot(constr_vals,constr_vals);

        if (constr_valsq > 0) {
            ret = opt_objfn(vals_inp,&grad_obj,opt_data) + c_pen*(constr_valsq / 2.0);

            if (grad_out) {
                *grad_out = grad_obj + c_pen*arma::trans(arma::sum(jacob_constr,0));
            }
        } else {
            ret = opt_objfn(vals_inp,&grad_obj,opt_data);

            if (grad_out) {
                *grad_out = grad_obj;
            }
        }
        //
        return ret;
    };

    //
    // initialization
    
    arma::vec x = init_out_vals;

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
        bfgs(x_p,sumt_objfn,&sumt_data);
        if (iter % 10 == 0) {
            err = arma::norm(x_p - x,2);
        }
        //
        sumt_data.c_pen = par_eta*sumt_data.c_pen; // increase penalization parameter value
        x = x_p;
    }
    //
    error_reporting(init_out_vals,x_p,opt_objfn,opt_data,success,value_out,err,err_tol,iter,iter_max,conv_failure_switch);
    //
    return success;
}

bool
optim::sumt(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data,
            std::function<arma::vec (const arma::vec& vals_inp, arma::mat* jacob_out, void* constr_data)> constr_fn, void* constr_data)
{
    return sumt_int(init_out_vals,opt_objfn,opt_data,constr_fn,constr_data,nullptr,nullptr);
}

bool
optim::sumt(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data,
            std::function<arma::vec (const arma::vec& vals_inp, arma::mat* jacob_out, void* constr_data)> constr_fn, void* constr_data,
            opt_settings& settings)
{
    return sumt_int(init_out_vals,opt_objfn,opt_data,constr_fn,constr_data,nullptr,&settings);
}

bool
optim::sumt(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data,
            std::function<arma::vec (const arma::vec& vals_inp, arma::mat* jacob_out, void* constr_data)> constr_fn, void* constr_data,
            double& value_out)
{
    return sumt_int(init_out_vals,opt_objfn,opt_data,constr_fn,constr_data,&value_out,nullptr);
}

bool
optim::sumt(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data,
            std::function<arma::vec (const arma::vec& vals_inp, arma::mat* jacob_out, void* constr_data)> constr_fn, void* constr_data,
            double& value_out, opt_settings& settings)
{
    return sumt_int(init_out_vals,opt_objfn,opt_data,constr_fn,constr_data,&value_out,&settings);
}
