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
 * Differential Evolution (DE) with Population Reduction and Multiple Mutation Strategies
 *
 * Keith O'Hara
 * 12/19/2016
 *
 * This version:
 * 08/05/2017
 */

#include "optim.hpp"

bool
optim::de_prmm_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, double* value_out, opt_settings* settings_inp)
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

    int n_pop = settings.de_n_pop;
    // const int check_freq = settings.de_check_freq;

    const double par_initial_F = settings.de_par_F;
    const double par_initial_CR = settings.de_par_CR;

    const arma::vec par_initial_lb = ((int) settings.de_lb.n_elem == n_vals) ? settings.de_lb : arma::zeros(n_vals,1) - 0.5;
    const arma::vec par_initial_ub = ((int) settings.de_ub.n_elem == n_vals) ? settings.de_ub : arma::zeros(n_vals,1) + 0.5;

    const double F_l = settings.de_par_F_l;
    const double F_u = settings.de_par_F_u;
    const double tau_F  = settings.de_par_tau_F;
    const double tau_CR = settings.de_par_tau_CR;

    arma::vec F_vec(n_pop), CR_vec(n_pop);
    F_vec.fill(par_initial_F);
    CR_vec.fill(par_initial_CR);

    const int max_fn_eval = settings.de_max_fn_eval;
    const int pmax = settings.de_pmax;
    const int n_pop_best = settings.de_n_pop_best;

    const double d_eps = 0.5;

    int n_gen = std::ceil(max_fn_eval / (pmax*n_pop));

    //
    // setup

    double prop_objfn_val = 0.0;
    arma::vec X_prop = init_out_vals, objfn_vals(n_pop);
    arma::mat X(n_pop,n_vals), X_next(n_pop,n_vals);

#ifdef OPTIM_OMP
    #pragma omp parallel for firstprivate(prop_objfn_val) 
#endif
    for (int i=0; i < n_pop; i++) {
        X_next.row(i) = init_out_vals.t() + par_initial_lb.t() + (par_initial_ub.t() - par_initial_lb.t())%arma::randu(1,n_vals);

        prop_objfn_val = opt_objfn(X_next.row(i).t(),nullptr,opt_data);

        if (!std::isfinite(prop_objfn_val)) {
            prop_objfn_val = BIG_POS_VAL;
        }
        
        objfn_vals(i) = prop_objfn_val;
    }

    double best_objfn_val_running = objfn_vals.min();
    // double best_objfn_val_check   = best_objfn_val_running;

    double best_val_main = best_objfn_val_running;
    double best_val_best = best_objfn_val_running;

    arma::vec best_sol_running = X_next.row( objfn_vals.index_min() ).t();
    arma::vec best_vec_main = best_sol_running;
    arma::vec best_vec_best = best_sol_running;

    arma::vec xchg_vec = best_sol_running;
    //
    int n_reset = 1;
    int iter = 0;
    double err = 2*err_tol;

    while (err > err_tol && iter < n_gen + 1) {
        iter++;

        //
        // population reduction step

        if (iter == n_gen && n_reset < 4) {
            int n_pop_temp = n_pop/2;

            arma::vec objfn_vals_reset(n_pop_temp);
            arma::mat X_reset(n_pop_temp,n_vals);

#ifdef OPTIM_OMP
            #pragma omp parallel for
#endif
            for (int j=0; j < n_pop_temp; j++) {
                if (objfn_vals(j) < objfn_vals(j + n_pop_temp)) {
                    X_reset.row(j) = X_next.row(j);
                    objfn_vals_reset(j) = objfn_vals(j);
                } else {
                    X_reset.row(j) = X_next.row(j + n_pop_temp);
                    objfn_vals_reset(j) = objfn_vals(j + n_pop_temp);
                }
            }

            objfn_vals = objfn_vals_reset;
            X_next = X_reset;

            n_pop /= 2;
            n_gen *= 2;

            iter = 1;
            n_reset++;
        }

        X = X_next;

        //
        // first population: n_pop - n_pop_best
        

#ifdef OPTIM_OMP
        #pragma omp parallel for firstprivate(prop_objfn_val,X_prop) 
#endif
        for (int i=0; i < n_pop - n_pop_best; i++) {

            arma::vec rand_pars = arma::randu(4);

            if (rand_pars(0) < tau_F) {
                F_vec(i) = F_l + (F_u-F_l)*rand_pars(1);
            }

            if (rand_pars(2) < tau_CR) {
                CR_vec(i) = rand_pars(3);
            }

            //

            int c_1, c_2, c_3;

            do {
                c_1 = arma::as_scalar(arma::randi(1, arma::distr_param(0, n_pop-1)));
            } while(c_1==i);

            do {
                c_2 = arma::as_scalar(arma::randi(1, arma::distr_param(0, n_pop-1)));
            } while(c_2==i || c_2==c_1);

            do {
                c_3 = arma::as_scalar(arma::randi(1, arma::distr_param(0, n_pop-1)));
            } while(c_3==i || c_3==c_1 || c_3==c_2);

            //

            int j = arma::as_scalar(arma::randi(1, arma::distr_param(0, n_vals-1)));

            arma::vec rand_unif = arma::randu(n_vals);

            for (int k=0; k < n_vals; k++) {
                if ( rand_unif(k) < CR_vec(i) || k == j ) {
                    double r_s = arma::as_scalar(arma::randu(1));

                    if ( r_s < 0.75 || n_pop >= 100 ) {
                        X_prop(k) = X(c_3,k) + F_vec(i)*(X(c_1,k) - X(c_2,k));
                    } else {
                        X_prop(k) = best_vec_main(k) + F_vec(i)*(X(c_1,k) - X(c_2,k));
                    }
                } else {
                    X_prop(k) = X(i,k);
                }
            }

            //

            prop_objfn_val = opt_objfn(X_prop,nullptr,opt_data);
            
            if (prop_objfn_val <= objfn_vals(i)) {
                X_next.row(i) = X_prop.t();
                objfn_vals(i) = prop_objfn_val;
            } else {
                X_next.row(i) = X.row(i);
            }
        }

        best_val_main = objfn_vals.rows(0,n_pop - n_pop_best - 1).min();
        best_vec_main = X_next.rows(0,n_pop - n_pop_best - 1).row( objfn_vals.rows(0,n_pop - n_pop_best - 1).index_min() ).t();

        if (best_val_main < best_val_best) {
            xchg_vec = best_vec_main;
        }

        //
        // second population

        for (int i = n_pop - n_pop_best; i < n_pop; i++) {

            arma::vec rand_pars = arma::randu(4);

            if (rand_pars(0) < tau_F) {
                F_vec(i) = F_l + (F_u-F_l)*rand_pars(1);
            }

            if (rand_pars(2) < tau_CR) {
                CR_vec(i) = rand_pars(3);
            }

            //

            int c_1, c_2;

            do {
                c_1 = arma::as_scalar(arma::randi(1, arma::distr_param(0, n_pop-1)));
            } while(c_1==i);

            do {
                c_2 = arma::as_scalar(arma::randi(1, arma::distr_param(0, n_pop-1)));
            } while(c_2==i || c_2==c_1);

            //

            int j = arma::as_scalar(arma::randi(1, arma::distr_param(0, n_vals-1)));

            arma::vec rand_unif = arma::randu(n_vals);

            for (int k=0; k < n_vals; k++) {
                if ( rand_unif(k) < CR_vec(i) || k == j ) {
                    X_prop(k) = best_vec_best(k) + F_vec(i)*(X(c_1,k) - X(c_2,k));
                } else {
                    X_prop(k) = X(i,k);
                }
            }

            //

            prop_objfn_val = opt_objfn(X_prop,nullptr,opt_data);
            
            if (prop_objfn_val <= objfn_vals(i)) {
                X_next.row(i) = X_prop.t();
                objfn_vals(i) = prop_objfn_val;
            } else {
                X_next.row(i) = X.row(i);
            }
        }

        best_val_best = objfn_vals.rows(n_pop - n_pop_best, n_pop - 1).min();
        best_vec_best = X_next.rows(n_pop - n_pop_best, n_pop - 1).row( objfn_vals.rows(n_pop - n_pop_best, n_pop - 1).index_min() ).t();

        if (best_val_best < best_val_main) {
            double the_sum = 0;

            for (int j=0; j < n_vals; j++) {
                double min_val = X.col(j).min();

                the_sum += (best_vec_best(j) - min_val) / (xchg_vec(j) - min_val);
            }

            the_sum /= (double) n_vals;

            if (std::abs(the_sum - 1.0) > d_eps) {
                best_vec_main = best_vec_best;
            } else {
                best_vec_best = best_vec_main;
            }
        } else {
            best_vec_best = best_vec_main;
        }

        //
        // assign running global minimum

        if (objfn_vals.min() < best_objfn_val_running) {
            best_objfn_val_running = objfn_vals.min();
            best_sol_running = X_next.row( objfn_vals.index_min() ).t();
        }

        // if (iter%check_freq == 0) {
        //     // err = std::abs(objfn_vals.min() - best_objfn_val) / (std::abs(best_objfn_val) + 1E-08);
        //     err = std::abs(best_objfn_val_running - best_objfn_val_check);
        //     // best_objfn_val_check = objfn_vals.min();
        //     if (best_objfn_val_running < best_objfn_val_check) {
        //         best_objfn_val_check = best_objfn_val_running;
        //     }
        // }
    }
    //
    error_reporting(init_out_vals,best_sol_running,opt_objfn,opt_data,success,value_out,err,err_tol,iter,n_gen,conv_failure_switch);
    //
    return success;
}

bool
optim::de_prmm(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data)
{
    return de_prmm_int(init_out_vals,opt_objfn,opt_data,nullptr,nullptr);
}

bool
optim::de_prmm(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, opt_settings& settings)
{
    return de_prmm_int(init_out_vals,opt_objfn,opt_data,nullptr,&settings);
}

bool
optim::de_prmm(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, double& value_out)
{
    return de_prmm_int(init_out_vals,opt_objfn,opt_data,&value_out,nullptr);
}

bool
optim::de_prmm(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, double& value_out, opt_settings& settings)
{
    return de_prmm_int(init_out_vals,opt_objfn,opt_data,&value_out,&settings);
}
