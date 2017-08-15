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
 * 08/14/2017
 */

#include "optim.hpp"

bool
optim::nm_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, opt_settings* settings_inp)
{
    bool success = false;

    const int n_vals = init_out_vals.n_elem;

    //
    // NM settings

    opt_settings settings;

    if (settings_inp) {
        settings = *settings_inp;
    }
    
    const int conv_failure_switch = settings.conv_failure_switch;
    const int iter_max = settings.iter_max;
    const double err_tol = settings.err_tol;

    // expansion / contraction parameters
    const double par_alpha = settings.nm_par_alpha;
    const double par_beta  = (settings.nm_adaptive) ? 0.75 - 1.0 / (2.0*n_vals) : settings.nm_par_beta;
    const double par_gamma = (settings.nm_adaptive) ? 1.0 + 2.0 / n_vals        : settings.nm_par_gamma;
    const double par_delta = (settings.nm_adaptive) ? 1.0 - 1.0 / n_vals        : settings.nm_par_delta;

    const bool vals_bound = settings.vals_bound;
    
    const arma::vec lower_bounds = settings.lower_bounds;
    const arma::vec upper_bounds = settings.upper_bounds;

    const arma::uvec bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    // lambda function for box constraints

    std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* box_data)> box_objfn = [opt_objfn, vals_bound, bounds_type, lower_bounds, upper_bounds] (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data) -> double {
        //

        if (vals_bound) {
            arma::vec vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);
            
            return opt_objfn(vals_inv_trans,nullptr,opt_data);
        } else {
            return opt_objfn(vals_inp,nullptr,opt_data);
        }
    };
    
    //
    // setup

    arma::vec simplex_fn_vals(n_vals+1);
    arma::mat simplex_points(n_vals+1,n_vals);
    
    simplex_fn_vals(0) = opt_objfn(init_out_vals,nullptr,opt_data);
    simplex_points.row(0) = init_out_vals.t();

    // for (int i=1; i < n_vals + 1; i++) {
    //     simplex_points.row(i) = init_out_vals.t() + 0.05*arma::trans(unit_vec(i-1,n_vals));
    //     simplex_fn_vals(i) = opt_objfn(simplex_points.row(i).t(),nullptr,opt_data);
    // }

    for (int i=1; i < n_vals + 1; i++) {
        if (init_out_vals(i-1) != 0.0) {
            simplex_points.row(i) = init_out_vals.t() + 0.05*init_out_vals(i-1)*arma::trans(unit_vec(i-1,n_vals));
        } else {
            simplex_points.row(i) = init_out_vals.t() + 0.00025*arma::trans(unit_vec(i-1,n_vals));
            // simplex_points.row(i) = init_out_vals.t() + 0.05*arma::trans(unit_vec(i-1,n_vals));
        }

        simplex_fn_vals(i) = opt_objfn(simplex_points.row(i).t(),nullptr,opt_data);

        if (vals_bound) {
            simplex_points.row(i) = arma::trans( transform(simplex_points.row(i).t(), bounds_type, lower_bounds, upper_bounds) );
        }
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

        x_r = centroid + par_alpha*(centroid - simplex_points.row(n_vals).t());

        f_r = box_objfn(x_r,nullptr,opt_data);

        if (f_r >= simplex_fn_vals(0) && f_r < simplex_fn_vals(n_vals-1)) {
            // reflected point is neither best nor worst in the new simplex
            simplex_points.row(n_vals) = x_r.t();
            next_iter = true;
        }

        // step 3

        if (!next_iter && f_r < simplex_fn_vals(0)) {
            // reflected point is better than the current best; try to go farther along this direction
            x_e = centroid + par_gamma*(x_r - centroid);

            f_e = box_objfn(x_e,nullptr,opt_data);

            if (f_e < f_r) {
                simplex_points.row(n_vals) = x_e.t();
            } else {
                simplex_points.row(n_vals) = x_r.t();
            }

            next_iter = true;
        }

        // steps 4, 5, 6

        if (!next_iter && f_r >= simplex_fn_vals(n_vals-1)) {
            // reflected point is still worse than x_n; contract

            // step 4, 5

            if (f_r < simplex_fn_vals(n_vals)) {
                // outside contraction
                x_oc = centroid + par_beta*(x_r - centroid);

                f_oc = box_objfn(x_oc,nullptr,opt_data);

                if (f_oc <= f_r) {
                    simplex_points.row(n_vals) = x_oc.t();
                    next_iter = true;
                }
            } else { // f_r >= simplex_fn_vals(n_vals)
                // inside contraction
                // x_ic = centroid - par_beta*(x_r - centroid);
                x_ic = centroid + par_beta*(simplex_points.row(n_vals).t() - centroid);

                f_ic = box_objfn(x_ic,nullptr,opt_data);

                if (f_ic < simplex_fn_vals(n_vals)) {
                    simplex_points.row(n_vals) = x_ic.t();
                    next_iter = true;
                }
            }
        }

        // step 6
        if (!next_iter) {
            // neither outside nor inside contraction was acceptable; shrink the simplex toward x(0)
            for (int i=1; i < n_vals + 1; i++) {
                simplex_points.row(i) = simplex_points.row(0) + par_delta*(simplex_points.row(i) - simplex_points.row(0));
            }
        }

        // check change in fn_val

        for (int i=0; i < n_vals + 1; i++) {
            simplex_fn_vals(i) = box_objfn(simplex_points.row(i).t(),nullptr,opt_data);
        }
    
        err = std::abs(min_val - simplex_fn_vals.max());
        min_val = simplex_fn_vals.min();
        iter++;
    }
    //
    arma::vec prop_out = simplex_points.row(index_min(simplex_fn_vals)).t();
    
    if (vals_bound) {
	    prop_out = inv_transform(prop_out, bounds_type, lower_bounds, upper_bounds);
    }

    error_reporting(init_out_vals,prop_out,opt_objfn,opt_data,success,err,err_tol,iter,iter_max,conv_failure_switch,settings_inp);
    //
    return success;
}

bool
optim::nm(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data)
{
    return nm_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

bool
optim::nm(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, opt_settings& settings)
{
    return nm_int(init_out_vals,opt_objfn,opt_data,&settings);
}
