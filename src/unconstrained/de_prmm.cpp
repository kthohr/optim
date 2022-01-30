/*################################################################################
  ##
  ##   Copyright (C) 2016-2022 Keith O'Hara
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

// [OPTIM_BEGIN]
optimlib_inline
bool
optim::internal::de_prmm_impl(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t* settings_inp
)
{
    bool success = false;

    const size_t n_vals = BMO_MATOPS_SIZE(init_out_vals);

    //
    // DE settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int print_level = settings.print_level;

    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const fp_t rel_objfn_change_tol = settings.rel_objfn_change_tol;

    size_t n_pop = settings.de_settings.n_pop;
    // const size_t check_freq = settings.de_settings.check_freq;

    const fp_t par_initial_F = settings.de_settings.par_F;
    const fp_t par_initial_CR = settings.de_settings.par_CR;

    const fp_t F_l = settings.de_settings.par_F_l;
    const fp_t F_u = settings.de_settings.par_F_u;
    const fp_t tau_F  = settings.de_settings.par_tau_F;
    const fp_t tau_CR = settings.de_settings.par_tau_CR;

    ColVec_t F_vec(n_pop), CR_vec(n_pop);
    F_vec.fill(par_initial_F);
    CR_vec.fill(par_initial_CR);

    const uint_t max_fn_eval = settings.de_settings.max_fn_eval;
    const uint_t pmax = settings.de_settings.pmax;
    const size_t n_pop_best = settings.de_settings.n_pop_best;

    const fp_t d_eps = settings.de_settings.par_d_eps;

    size_t n_gen = std::ceil(max_fn_eval / (pmax*n_pop));
    const size_t check_freq = settings.de_settings.check_freq;

    const bool return_population_mat = settings.de_settings.return_population_mat;

    const bool vals_bound = settings.vals_bound;
    
    const ColVec_t lower_bounds = settings.lower_bounds;
    const ColVec_t upper_bounds = settings.upper_bounds;

    const ColVecInt_t bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    ColVec_t par_initial_lb = ( BMO_MATOPS_SIZE(settings.de_settings.initial_lb) == n_vals ) ? settings.de_settings.initial_lb : BMO_MATOPS_ARRAY_ADD_SCALAR(init_out_vals, -0.5);
    ColVec_t par_initial_ub = ( BMO_MATOPS_SIZE(settings.de_settings.initial_ub) == n_vals ) ? settings.de_settings.initial_ub : BMO_MATOPS_ARRAY_ADD_SCALAR(init_out_vals,  0.5);

    sampling_bounds_check(vals_bound, n_vals, bounds_type, lower_bounds, upper_bounds, par_initial_lb, par_initial_ub);

    // random sampling setup

    int omp_n_threads = 1;
    rand_engine_t rand_engine(settings.rng_seed_value);
    std::vector<rand_engine_t> engines;

#ifdef OPTIM_USE_OMP
    if (settings.de_settings.omp_n_threads > 0) {
        omp_n_threads = settings.de_settings.omp_n_threads;
    } else {
        omp_n_threads = std::max(1, static_cast<int>(omp_get_max_threads()) / 2); // OpenMP often detects the number of virtual/logical cores, not physical cores
    }
#endif

    for (int i = 0; i < omp_n_threads; ++i) {
        size_t seed_val = generate_seed_value(i, omp_n_threads, rand_engine);
        engines.push_back(rand_engine_t(seed_val));
    }

    // lambda function for box constraints

    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* box_data)> box_objfn \
    = [opt_objfn, vals_bound, bounds_type, lower_bounds, upper_bounds] (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data) \
    -> fp_t 
    {
        if (vals_bound) {
            ColVec_t vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);
            
            return opt_objfn(vals_inv_trans, nullptr, opt_data);
        } else {
            return opt_objfn(vals_inp, nullptr, opt_data);
        }
    };

    //
    // setup

    ColVec_t rand_vec(n_vals);
    ColVec_t objfn_vals(n_pop);
    Mat_t X(n_pop,n_vals), X_next(n_pop,n_vals);

#ifdef OPTIM_USE_OMP
    #pragma omp parallel for num_threads(omp_n_threads) private(rand_vec)
#endif
    for (size_t i = 0; i < n_pop; ++i) {
        size_t thread_num = 0;

#ifdef OPTIM_USE_OMP
        thread_num = omp_get_thread_num();
#endif

        bmo_stats::internal::runif_vec_inplace<fp_t>(n_vals, engines[thread_num], rand_vec);

        X_next.row(i) = BMO_MATOPS_TRANSPOSE( par_initial_lb + BMO_MATOPS_HADAMARD_PROD( (par_initial_ub - par_initial_lb), rand_vec ) );

        fp_t prop_objfn_val = opt_objfn( BMO_MATOPS_TRANSPOSE(X_next.row(i)), nullptr, opt_data);

        if (!std::isfinite(prop_objfn_val)) {
            prop_objfn_val = inf;
        }
        
        objfn_vals(i) = prop_objfn_val;

        if (vals_bound) {
            X_next.row(i) = transform<RowVec_t>( X_next.row(i), bounds_type, lower_bounds, upper_bounds);
        }
    }

    size_t min_objfn_val_index = index_min(objfn_vals);
    fp_t min_objfn_val = objfn_vals(min_objfn_val_index);

    fp_t min_objfn_val_running = min_objfn_val;
    fp_t min_objfn_val_check   = min_objfn_val_running;

    fp_t best_val_main = min_objfn_val_running;
    fp_t best_val_best = min_objfn_val_running;

    RowVec_t best_sol_running = X_next.row( min_objfn_val_index );
    RowVec_t best_vec_main = best_sol_running;
    RowVec_t best_vec_best = best_sol_running;

    RowVec_t xchg_vec = best_sol_running;

    //

    uint_t n_reset = 1;
    size_t iter = 0;
    fp_t rel_objfn_change = 2*rel_objfn_change_tol;
    ColVec_t rand_pars(4);

    while (rel_objfn_change > rel_objfn_change_tol && iter < n_gen + 1) {
        ++iter;

        //
        // population reduction step

        if (iter == n_gen && n_reset < 4) {
            size_t n_pop_temp = n_pop/2;

            ColVec_t objfn_vals_reset(n_pop_temp);
            Mat_t X_reset(n_pop_temp,n_vals);

#ifdef OPTIM_USE_OMP
            #pragma omp parallel for num_threads(omp_n_threads)
#endif
            for (size_t j = 0; j < n_pop_temp; ++j) {
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
            ++n_reset;
        }

        X = X_next;

        //
        // first population: n_pop - n_pop_best
        
#ifdef OPTIM_USE_OMP
        #pragma omp parallel for num_threads(omp_n_threads) private(rand_vec,rand_pars)
#endif
        for (size_t i = 0; i < n_pop - n_pop_best; ++i) {
            size_t thread_num = 0;

#ifdef OPTIM_USE_OMP
            thread_num = omp_get_thread_num();
#endif

            bmo_stats::internal::runif_vec_inplace<fp_t>(4, engines[thread_num], rand_pars);

            if (rand_pars(0) < tau_F) {
                F_vec(i) = F_l + (F_u-F_l) * rand_pars(1);
            }

            if (rand_pars(2) < tau_CR) {
                CR_vec(i) = rand_pars(3);
            }

            //

            uint_t c_1, c_2, c_3;

            do {
                c_1 = bmo_stats::rind(0, n_pop-1, engines[thread_num]);
            } while (c_1 == i);

            do {
                c_2 = bmo_stats::rind(0, n_pop-1, engines[thread_num]);
            } while (c_2==i || c_2==c_1);

            do {
                c_3 = bmo_stats::rind(0, n_pop-1, engines[thread_num]);
            } while (c_3==i || c_3==c_1 || c_3==c_2);

            //

            const size_t rand_ind = bmo_stats::rind(0, n_vals-1, engines[thread_num]);

            // ColVec_t rand_unif = bmo_stats::runif_vec<fp_t>(n_vals, engines[thread_num]);
            bmo_stats::internal::runif_vec_inplace<fp_t>(n_vals, engines[thread_num], rand_vec);
            RowVec_t X_prop(n_vals);

            for (size_t k = 0; k < n_vals; ++k) {
                if ( rand_vec(k) < CR_vec(i) || k == rand_ind ) {
                    fp_t r_s = bmo_stats::runif<fp_t>(engines[thread_num]);

                    if ( r_s < fp_t(0.75) || n_pop >= 100 ) {
                        X_prop(k) = X(c_3,k) + F_vec(i) * (X(c_1,k) - X(c_2,k));
                    } else {
                        X_prop(k) = best_vec_main(k) + F_vec(i) * (X(c_1,k) - X(c_2,k));
                    }
                } else {
                    X_prop(k) = X(i,k);
                }
            }

            //

            fp_t prop_objfn_val = box_objfn( BMO_MATOPS_TRANSPOSE(X_prop),nullptr,opt_data);
            
            if (prop_objfn_val <= objfn_vals(i)) {
                X_next.row(i) = X_prop;
                objfn_vals(i) = prop_objfn_val;
            } else {
                X_next.row(i) = X.row(i);
            }
        }

        min_objfn_val_index = index_min( BMO_MATOPS_MIDDLE_ROWS(objfn_vals, 0, n_pop - n_pop_best - 1) );

        best_val_main = objfn_vals(min_objfn_val_index);
        // best_vec_main = X_next.rows(0,n_pop - n_pop_best - 1).row( objfn_vals.rows(0,n_pop - n_pop_best - 1).index_min() );
        best_vec_main = X_next.row( min_objfn_val_index );

        if (best_val_main < best_val_best) {
            xchg_vec = best_vec_main;
        }

        //
        // second population

#ifdef OPTIM_USE_OMP
        #pragma omp parallel for num_threads(omp_n_threads) private(rand_vec,rand_pars)
#endif
        for (size_t i = n_pop - n_pop_best; i < n_pop; ++i) {
            size_t thread_num = 0;

#ifdef OPTIM_USE_OMP
            thread_num = omp_get_thread_num();
#endif

            bmo_stats::internal::runif_vec_inplace<fp_t>(4, engines[thread_num], rand_pars);

            if (rand_pars(0) < tau_F) {
                F_vec(i) = F_l + (F_u-F_l) * rand_pars(1);
            }

            if (rand_pars(2) < tau_CR) {
                CR_vec(i) = rand_pars(3);
            }

            //

            uint_t c_1, c_2;

            do {
                c_1 = bmo_stats::rind(0, n_pop-1, engines[thread_num]);
            } while(c_1 == i);

            do {
                c_2 = bmo_stats::rind(0, n_pop-1, engines[thread_num]);
            } while(c_2==i || c_2==c_1);

            //

            const size_t rand_ind = bmo_stats::rind(0, n_vals-1, engines[thread_num]);

            bmo_stats::internal::runif_vec_inplace<fp_t>(n_vals, engines[thread_num], rand_vec);
            RowVec_t X_prop(n_vals);

            for (size_t k = 0; k < n_vals; ++k) {
                if ( rand_vec(k) < CR_vec(i) || k == rand_ind ) {
                    X_prop(k) = best_vec_best(k) + F_vec(i) * (X(c_1,k) - X(c_2,k));
                } else {
                    X_prop(k) = X(i,k);
                }
            }

            //

            fp_t prop_objfn_val = box_objfn( BMO_MATOPS_TRANSPOSE(X_prop), nullptr, opt_data);
            
            if (prop_objfn_val <= objfn_vals(i)) {
                X_next.row(i) = X_prop;
                objfn_vals(i) = prop_objfn_val;
            } else {
                X_next.row(i) = X.row(i);
            }
        }

        min_objfn_val_index = n_pop - n_pop_best + index_min( BMO_MATOPS_MIDDLE_ROWS(objfn_vals, n_pop - n_pop_best, n_pop - 1) );

        // best_val_best = objfn_vals.rows(n_pop - n_pop_best, n_pop - 1).min();
        best_val_best = objfn_vals(min_objfn_val_index);
        // best_vec_best = X_next.rows(n_pop - n_pop_best, n_pop - 1).row( objfn_vals.rows(n_pop - n_pop_best, n_pop - 1).index_min() );
        best_vec_best = X_next.row( min_objfn_val_index );

        if (best_val_best < best_val_main) {
            fp_t the_sum = 0.0;

            for (size_t j = 0; j < n_vals; ++j) {
                fp_t min_val = BMO_MATOPS_MIN_VAL(X.col(j));

                the_sum += (best_vec_best(j) - min_val) / (xchg_vec(j) - min_val);
            }

            the_sum /= static_cast<fp_t>(n_vals);

            if (std::abs(the_sum - 1.0) > d_eps) {
                best_vec_main = best_vec_best;
            } else {
                best_vec_best = best_vec_main;
            }
        } else {
            best_vec_best = best_vec_main;
        }

        // assign running global minimum

        min_objfn_val_index = index_min( BMO_MATOPS_MIDDLE_ROWS(objfn_vals, 0, n_pop - 1) );
        fp_t best_val_tmp = objfn_vals(min_objfn_val_index);

        if (best_val_tmp < min_objfn_val_running) {
            min_objfn_val_running = best_val_tmp;
            best_sol_running = X_next.row( min_objfn_val_index );
        }

        if (iter % check_freq == 0) {   
            rel_objfn_change = std::abs(min_objfn_val_running - min_objfn_val_check) / (OPTIM_FPN_SMALL_NUMBER + std::abs(min_objfn_val_running));
            
            if (min_objfn_val_running < min_objfn_val_check) {
                min_objfn_val_check = min_objfn_val_running;
            }
        }

        OPTIM_DEPRMM_TRACE(iter, rel_objfn_change, min_objfn_val_running, min_objfn_val_check, best_sol_running, X_next);
    }

    //

    if (return_population_mat) {
        if (vals_bound) {
#ifdef OPTIM_USE_OMP
        #pragma omp parallel for num_threads(omp_n_threads)
#endif
            for (size_t i = 0; i < n_pop; ++i) {
                X_next.row(i) = inv_transform<RowVec_t>(X_next.row(i), bounds_type, lower_bounds, upper_bounds);
            }
        }

        settings_inp->de_settings.population_mat = X_next;
    }

    //

    if (vals_bound) {
        best_sol_running = inv_transform(best_sol_running, bounds_type, lower_bounds, upper_bounds);
    }

    error_reporting(init_out_vals, BMO_MATOPS_TRANSPOSE(best_sol_running), opt_objfn, opt_data, 
                    success, rel_objfn_change, rel_objfn_change_tol, iter, n_gen, 
                    conv_failure_switch, settings_inp);

    //

    return true;
}

optimlib_inline
bool
optim::de_prmm(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data
)
{
    return internal::de_prmm_impl(init_out_vals,opt_objfn,opt_data,nullptr);
}

optimlib_inline
bool
optim::de_prmm(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t& settings
)
{
    return internal::de_prmm_impl(init_out_vals,opt_objfn,opt_data,&settings);
}
