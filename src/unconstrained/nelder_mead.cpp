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
 * Nelder-Mead
 *
 * Keith O'Hara
 * 01/03/2017
 *
 * This version:
 * 07/19/2017
 */

#include "optim.hpp"

bool
optim::nelder_mead_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                       double* value_out, optim_opt_settings* opt_params)
{
    bool success = false;
    
    const int conv_failure_switch = (opt_params) ? opt_params->conv_failure_switch : OPTIM_CONV_FAILURE_POLICY;
    const int iter_max   = (opt_params) ? opt_params->iter_max : OPTIM_DEFAULT_ITER_MAX;
    const double err_tol = (opt_params) ? opt_params->err_tol  : OPTIM_DEFAULT_ERR_TOL;

    // expansion / contraction parameters
    const double alpha = (opt_params) ? opt_params->alpha_nm : OPTIM_DEFAULT_NM_ALPHA;
    const double beta  = (opt_params) ? opt_params->beta_nm  : OPTIM_DEFAULT_NM_BETA;
    const double gamma = (opt_params) ? opt_params->gamma_nm : OPTIM_DEFAULT_NM_GAMMA;
    const double delta = (opt_params) ? opt_params->delta_nm : OPTIM_DEFAULT_NM_DELTA;
    //
    const int n_vals = init_out_vals.n_elem;
    arma::vec simplex_fn_vals(n_vals+1);
    arma::mat simplex_points(n_vals+1,n_vals);
    
    simplex_fn_vals(0) = opt_objfn(init_out_vals,nullptr,opt_data);
    simplex_points.row(0) = init_out_vals.t();

    for (int i=1; i < n_vals + 1; i++) {
        simplex_points.row(i) = init_out_vals.t() + 0.05*arma::trans(unit_vec(i-1,n_vals));
        simplex_fn_vals(i) = opt_objfn(simplex_points.row(i).t(),nullptr,opt_data);
    }

    double min_val = simplex_fn_vals.min();
    //
    int iter = 0;
    double err = 2*err_tol;
    arma::uvec sort_vec;

    double f_r, f_e, f_oc, f_ic;
    arma::vec centroid, x_r, x_e, x_oc, x_ic;

    while (err > err_tol && iter < iter_max) {

        bool next_iter = false;
        
        // step 1

        sort_vec = arma::sort_index(simplex_fn_vals);

        simplex_fn_vals = simplex_fn_vals(sort_vec);
        simplex_points = simplex_points.rows(sort_vec);

        // step 2

        centroid = arma::trans(arma::sum(simplex_points.rows(0,n_vals-1),0)) / (double) n_vals;

        x_r = centroid + alpha*(centroid - simplex_points.row(n_vals).t());

        f_r = opt_objfn(x_r,nullptr,opt_data);

        if (f_r >= simplex_fn_vals(0) && f_r < simplex_fn_vals(n_vals-1)) {
            // reflected point is neither best nor worst in the new simplex
            simplex_points.row(n_vals) = x_r.t();
            next_iter = true;
        }

        // step 3

        if (!next_iter && f_r < simplex_fn_vals(0)) {
            // reflected point is better than the current best; try to go farther along this direction
            x_e = centroid + beta*(x_r - centroid);

            f_e = opt_objfn(x_e,nullptr,opt_data);

            if (f_e < f_r) {
                simplex_points.row(n_vals) = x_e.t();
            } else {
                simplex_points.row(n_vals) = x_r.t();
            }

            next_iter = true;
        }

        // steps 4, 5, 6

        if (!next_iter && f_r >= simplex_fn_vals(n_vals-1)) {
            // reflected point is still worse than xn; contract

            // step 4, 5

            if (f_r < simplex_fn_vals(n_vals)) {
                // outside contraction
                x_oc = centroid + gamma*(x_r - centroid);

                f_oc = opt_objfn(x_oc,nullptr,opt_data);

                if (f_oc <= f_r) {
                    simplex_points.row(n_vals) = x_oc.t();
                    next_iter = true;
                }
            } else {
                // inside contraction
                x_ic = centroid - gamma*(x_r - centroid);

                f_ic = opt_objfn(x_ic,nullptr,opt_data);

                if (f_ic < simplex_fn_vals(n_vals)) {
                    simplex_points.row(n_vals) = x_ic.t();
                    next_iter = true;
                }
            }

            // step 6
            if (!next_iter) {
                // neither outside nor inside contraction was acceptable; shrink the simplex toward x(0)
                for (int i=1; i < n_vals + 1; i++) {
                    simplex_points.row(i) = simplex_points.row(0) + delta*(simplex_points.row(i) - simplex_points.row(0));
                }
            }

        }

        // check change in fn_val

        for (int i=0; i < n_vals + 1; i++) {
            simplex_fn_vals(i) = opt_objfn(simplex_points.row(i).t(),nullptr,opt_data);
        }

        err = std::abs(min_val - simplex_fn_vals.max());
        min_val = simplex_fn_vals.min();
        iter++;
    }
    //
    arma::vec prop_out = simplex_points.row(index_min(simplex_fn_vals)).t();
    error_reporting(init_out_vals,prop_out,opt_objfn,opt_data,success,value_out,err,err_tol,iter,iter_max,conv_failure_switch);
    //
    return success;
}

bool
optim::nelder_mead(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data)
{
    return nelder_mead_int(init_out_vals,opt_objfn,opt_data,nullptr,nullptr);
}

bool
optim::nelder_mead(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data, optim_opt_settings& opt_params)
{
    return nelder_mead_int(init_out_vals,opt_objfn,opt_data,nullptr,&opt_params);
}

bool
optim::nelder_mead(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data, double& value_out)
{
    return nelder_mead_int(init_out_vals,opt_objfn,opt_data,&value_out,nullptr);
}

bool
optim::nelder_mead(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data, double& value_out, optim_opt_settings& opt_params)
{
    return nelder_mead_int(init_out_vals,opt_objfn,opt_data,&value_out,&opt_params);
}
