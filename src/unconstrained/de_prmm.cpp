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
 * Differential Evolution (DE) with Population Reduction and Multiple Mutation Strategies
 */

#include "optim.hpp"

bool
optim::de_prmm_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t* settings_inp)
{
    bool success = false;

    const size_t n_vals = init_out_vals.n_elem;

    //
    // DE settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const double err_tol = settings.err_tol;

    size_t n_pop = settings.de_n_pop;
    // const size_t check_freq = settings.de_check_freq;

    const double par_initial_F = settings.de_par_F;
    const double par_initial_CR = settings.de_par_CR;

    const arma::vec par_initial_lb = (settings.de_initial_lb.n_elem == n_vals) ? settings.de_initial_lb : init_out_vals - 0.5;
    const arma::vec par_initial_ub = (settings.de_initial_ub.n_elem == n_vals) ? settings.de_initial_ub : init_out_vals + 0.5;

    const double F_l = settings.de_par_F_l;
    const double F_u = settings.de_par_F_u;
    const double tau_F  = settings.de_par_tau_F;
    const double tau_CR = settings.de_par_tau_CR;

    arma::vec F_vec(n_pop), CR_vec(n_pop);
    F_vec.fill(par_initial_F);
    CR_vec.fill(par_initial_CR);

    const uint_t max_fn_eval = settings.de_max_fn_eval;
    const uint_t pmax = settings.de_pmax;
    const size_t n_pop_best = settings.de_n_pop_best;

    const double d_eps = 0.5;

    size_t n_gen = std::ceil(max_fn_eval / (pmax*n_pop));

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
            
            return opt_objfn(vals_inv_trans,nullptr,opt_data);
        }
        else
        {
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
    for (size_t i=0; i < n_pop; i++) 
    {
        X_next.row(i) = par_initial_lb.t() + (par_initial_ub.t() - par_initial_lb.t())%arma::randu(1,n_vals);

        double prop_objfn_val = opt_objfn(X_next.row(i).t(),nullptr,opt_data);

        if (!std::isfinite(prop_objfn_val)) {
            prop_objfn_val = inf;
        }
        
        objfn_vals(i) = prop_objfn_val;

        if (vals_bound) {
            X_next.row(i) = arma::trans( transform(X_next.row(i).t(), bounds_type, lower_bounds, upper_bounds) );
        }
    }

    double best_objfn_val_running = objfn_vals.min();
    // double best_objfn_val_check   = best_objfn_val_running;

    double best_val_main = best_objfn_val_running;
    double best_val_best = best_objfn_val_running;

    arma::rowvec best_sol_running = X_next.row( objfn_vals.index_min() );
    arma::rowvec best_vec_main = best_sol_running;
    arma::rowvec best_vec_best = best_sol_running;

    arma::rowvec xchg_vec = best_sol_running;

    //

    uint_t n_reset = 1;
    uint_t iter = 0;
    double err = 2*err_tol;

    while (err > err_tol && iter < n_gen + 1)
    {
        iter++;

        //
        // population reduction step

        if (iter == n_gen && n_reset < 4)
        {
            size_t n_pop_temp = n_pop/2;

            arma::vec objfn_vals_reset(n_pop_temp);
            arma::mat X_reset(n_pop_temp,n_vals);

#ifdef OPTIM_USE_OMP
            #pragma omp parallel for
#endif
            for (size_t j=0; j < n_pop_temp; j++) 
            {
                if (objfn_vals(j) < objfn_vals(j + n_pop_temp)) 
                {
                    X_reset.row(j) = X_next.row(j);
                    objfn_vals_reset(j) = objfn_vals(j);
                }
                else
                {
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
        

#ifdef OPTIM_USE_OMP
        #pragma omp parallel for
#endif
        for (size_t i=0; i < n_pop - n_pop_best; i++)
        {
            arma::vec rand_pars = arma::randu(4);

            if (rand_pars(0) < tau_F) {
                F_vec(i) = F_l + (F_u-F_l)*rand_pars(1);
            }

            if (rand_pars(2) < tau_CR) {
                CR_vec(i) = rand_pars(3);
            }

            //

            uint_t c_1, c_2, c_3;

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

            size_t j = arma::as_scalar(arma::randi(1, arma::distr_param(0, n_vals-1)));

            arma::vec rand_unif = arma::randu(n_vals);
            arma::rowvec X_prop(n_vals);

            for (size_t k=0; k < n_vals; k++)
            {
                if ( rand_unif(k) < CR_vec(i) || k == j )
                {
                    double r_s = arma::as_scalar(arma::randu(1));

                    if ( r_s < 0.75 || n_pop >= 100 ) {
                        X_prop(k) = X(c_3,k) + F_vec(i)*(X(c_1,k) - X(c_2,k));
                    } else {
                        X_prop(k) = best_vec_main(k) + F_vec(i)*(X(c_1,k) - X(c_2,k));
                    }
                }
                else
                {
                    X_prop(k) = X(i,k);
                }
            }

            //

            double prop_objfn_val = box_objfn(X_prop.t(),nullptr,opt_data);
            
            if (prop_objfn_val <= objfn_vals(i))
            {
                X_next.row(i) = X_prop;
                objfn_vals(i) = prop_objfn_val;
            }
            else
            {
                X_next.row(i) = X.row(i);
            }
        }

        best_val_main = objfn_vals.rows(0,n_pop - n_pop_best - 1).min();
        best_vec_main = X_next.rows(0,n_pop - n_pop_best - 1).row( objfn_vals.rows(0,n_pop - n_pop_best - 1).index_min() );

        if (best_val_main < best_val_best) {
            xchg_vec = best_vec_main;
        }

        //
        // second population

        for (size_t i = n_pop - n_pop_best; i < n_pop; i++)
        {
            arma::vec rand_pars = arma::randu(4);

            if (rand_pars(0) < tau_F) {
                F_vec(i) = F_l + (F_u-F_l)*rand_pars(1);
            }

            if (rand_pars(2) < tau_CR) {
                CR_vec(i) = rand_pars(3);
            }

            //

            uint_t c_1, c_2;

            do {
                c_1 = arma::as_scalar(arma::randi(1, arma::distr_param(0, n_pop-1)));
            } while(c_1==i);

            do {
                c_2 = arma::as_scalar(arma::randi(1, arma::distr_param(0, n_pop-1)));
            } while(c_2==i || c_2==c_1);

            //

            size_t j = arma::as_scalar(arma::randi(1, arma::distr_param(0, n_vals-1)));

            arma::vec rand_unif = arma::randu(n_vals);
            arma::rowvec X_prop(n_vals);

            for (size_t k=0; k < n_vals; k++) {
                if ( rand_unif(k) < CR_vec(i) || k == j ) {
                    X_prop(k) = best_vec_best(k) + F_vec(i)*(X(c_1,k) - X(c_2,k));
                } else {
                    X_prop(k) = X(i,k);
                }
            }

            //

            double prop_objfn_val = box_objfn(X_prop.t(),nullptr,opt_data);
            
            if (prop_objfn_val <= objfn_vals(i))
            {
                X_next.row(i) = X_prop;
                objfn_vals(i) = prop_objfn_val;
            }
            else
            {
                X_next.row(i) = X.row(i);
            }
        }

        best_val_best = objfn_vals.rows(n_pop - n_pop_best, n_pop - 1).min();
        best_vec_best = X_next.rows(n_pop - n_pop_best, n_pop - 1).row( objfn_vals.rows(n_pop - n_pop_best, n_pop - 1).index_min() );

        if (best_val_best < best_val_main)
        {
            double the_sum = 0;

            for (size_t j=0; j < n_vals; j++) 
            {
                double min_val = X.col(j).min();

                the_sum += (best_vec_best(j) - min_val) / (xchg_vec(j) - min_val);
            }

            the_sum /= static_cast<double>(n_vals);

            if (std::abs(the_sum - 1.0) > d_eps) {
                best_vec_main = best_vec_best;
            } else {
                best_vec_best = best_vec_main;
            }
        }
        else
        {
            best_vec_best = best_vec_main;
        }

        //
        // assign running global minimum

        if (objfn_vals.min() < best_objfn_val_running)
        {
            best_objfn_val_running = objfn_vals.min();
            best_sol_running = X_next.row( objfn_vals.index_min() );
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

    if (vals_bound) {
        best_sol_running = arma::trans( inv_transform(best_sol_running.t(), bounds_type, lower_bounds, upper_bounds) );
    }

    error_reporting(init_out_vals,best_sol_running.t(),opt_objfn,opt_data,success,err,err_tol,iter,n_gen,conv_failure_switch,settings_inp);

    //

    return true;
}

bool
optim::de_prmm(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data)
{
    return de_prmm_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

bool
optim::de_prmm(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t& settings)
{
    return de_prmm_int(init_out_vals,opt_objfn,opt_data,&settings);
}
