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
 * Differential Evolution (DE) optimization
 *
 * Keith O'Hara
 * 12/19/2016
 *
 * This version:
 * 08/14/2017
 */

#include "optim.hpp"

bool
optim::de_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, opt_settings* settings_inp)
{
    bool success = false;

    const double BIG_POS_VAL = OPTIM_BIG_POS_VAL;
    const int n_vals = init_out_vals.n_elem;

    //
    // DE settings

    opt_settings settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int conv_failure_switch = settings.conv_failure_switch;
    const double err_tol = settings.err_tol;

    const int n_pop = settings.de_n_pop;
    const int n_gen = settings.de_n_gen;
    const int check_freq = (settings.de_check_freq > 0) ? settings.de_check_freq : n_gen ;

    const int mutation_method = settings.de_mutation_method;

    const double par_F = settings.de_par_F;
    const double par_CR = settings.de_par_CR;

    const arma::vec par_initial_lb = ((int) settings.de_initial_lb.n_elem == n_vals) ? settings.de_initial_lb : init_out_vals - 0.5;
    const arma::vec par_initial_ub = ((int) settings.de_initial_ub.n_elem == n_vals) ? settings.de_initial_ub : init_out_vals + 0.5;

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

    arma::vec objfn_vals(n_pop);
    arma::mat X(n_pop,n_vals), X_next(n_pop,n_vals);

#ifdef OPTIM_USE_OMP
    #pragma omp parallel for
#endif
    for (int i=0; i < n_pop; i++) {
        X_next.row(i) = par_initial_lb.t() + (par_initial_ub.t() - par_initial_lb.t())%arma::randu(1,n_vals);

        double prop_objfn_val = opt_objfn(X_next.row(i).t(),nullptr,opt_data);

        if (!std::isfinite(prop_objfn_val)) {
            prop_objfn_val = BIG_POS_VAL;
        }
        
        objfn_vals(i) = prop_objfn_val;

        if (vals_bound) {
            X_next.row(i) = arma::trans( transform(X_next.row(i).t(), bounds_type, lower_bounds, upper_bounds) );
        }
    }

    double best_val = objfn_vals.min();
    double best_objfn_val_running = best_val;
    double best_objfn_val_check   = best_val;

    arma::rowvec best_vec = X_next.row( objfn_vals.index_min() );
    arma::rowvec best_sol_running = best_vec;

    //
    // begin loop

    int iter = 0;
    double err = 2*err_tol;

    while (err > err_tol && iter < n_gen + 1) {
        iter++;

        X = X_next;

        //
        // loop over population

#ifdef OPTIM_USE_OMP
        #pragma omp parallel for
#endif
        for (int i=0; i < n_pop; i++) {

            int c_1, c_2, c_3;

            do { // 'r_2' in paper's notation
                c_1 = arma::as_scalar(arma::randi(1, arma::distr_param(0, n_pop-1)));
            } while(c_1==i);

            do { // 'r_3' in paper's notation
                c_2 = arma::as_scalar(arma::randi(1, arma::distr_param(0, n_pop-1)));
            } while(c_2==i || c_2==c_1);

            do { // 'r_1' in paper's notation
                c_3 = arma::as_scalar(arma::randi(1, arma::distr_param(0, n_pop-1)));
            } while(c_3==i || c_3==c_1 || c_3==c_2);

            //

            int j = arma::as_scalar(arma::randi(1, arma::distr_param(0, n_vals-1)));

            arma::vec rand_unif = arma::randu(n_vals);
            arma::rowvec X_prop(n_vals);

            for (int k=0; k < n_vals; k++) {
                if ( rand_unif(k) < par_CR || k == j ) {

                    if ( mutation_method == 1 ) {
                        X_prop(k) = X(c_3,k) + par_F*(X(c_1,k) - X(c_2,k));
                    } else {
                        X_prop(k) = best_vec(k) + par_F*(X(c_1,k) - X(c_2,k));
                    }
                } else {
                    X_prop(k) = X(i,k);
                }
            }

            //

            double prop_objfn_val = box_objfn(X_prop.t(),nullptr,opt_data);

            if (!std::isfinite(prop_objfn_val)) {
                prop_objfn_val = BIG_POS_VAL;
            }
            
            if (prop_objfn_val <= objfn_vals(i)) {
                X_next.row(i) = X_prop;
                objfn_vals(i) = prop_objfn_val;
            } else {
                X_next.row(i) = X.row(i);
            }
        }

        best_val = objfn_vals.min();
        best_vec = X_next.row( objfn_vals.index_min() );

        //
        // assign running global minimum

        if (best_val < best_objfn_val_running) {
            best_objfn_val_running = objfn_vals.min();
            best_sol_running = X_next.row( objfn_vals.index_min() );
        }

        if (iter%check_freq == 0) {
            
            err = std::abs(best_objfn_val_running - best_objfn_val_check);
            
            if (best_objfn_val_running < best_objfn_val_check) {
                best_objfn_val_check = best_objfn_val_running;
            }
        }
    }
    //
    if (vals_bound) {
	    best_sol_running = arma::trans( inv_transform(best_sol_running.t(), bounds_type, lower_bounds, upper_bounds) );
    }

    error_reporting(init_out_vals,best_sol_running.t(),opt_objfn,opt_data,success,err,err_tol,iter,n_gen,conv_failure_switch,settings_inp);
    //
    return true;
}

bool
optim::de(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data)
{
    return de_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

bool
optim::de(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, opt_settings& settings)
{
    return de_int(init_out_vals,opt_objfn,opt_data,&settings);
}
