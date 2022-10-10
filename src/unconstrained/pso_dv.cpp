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
 * Particle Swarm Optimization (PSO) with Differentially-Perturbed Velocity (DV)
 */

#include "optim.hpp"

// [OPTIM_BEGIN]
optimlib_inline
bool
optim::internal::pso_dv_impl(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t* settings_inp
)
{
    bool success = false;

    const size_t n_vals = BMO_MATOPS_SIZE(init_out_vals);

    //
    // PSO settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int print_level = settings.print_level;

    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const fp_t rel_objfn_change_tol = settings.rel_objfn_change_tol;

    const size_t n_pop = settings.pso_settings.n_pop;
    const size_t n_gen = settings.pso_settings.n_gen;

    const size_t check_freq = settings.pso_settings.check_freq;

    const uint_t stag_limit = 50;

    fp_t par_w = 1.0;
    fp_t par_beta = 0.5;
    const fp_t par_damp = 0.99;
    // const fp_t par_c_1 = 1.494;
    const fp_t par_c_2 = 1.494;

    const fp_t par_CR = 0.7;

    const bool vals_bound = settings.vals_bound;
    
    const ColVec_t lower_bounds = settings.lower_bounds;
    const ColVec_t upper_bounds = settings.upper_bounds;

    const ColVecInt_t bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    ColVec_t par_initial_lb = ( BMO_MATOPS_SIZE(settings.pso_settings.initial_lb) == n_vals ) ? settings.pso_settings.initial_lb : BMO_MATOPS_ARRAY_ADD_SCALAR(init_out_vals, -0.5);
    ColVec_t par_initial_ub = ( BMO_MATOPS_SIZE(settings.pso_settings.initial_ub) == n_vals ) ? settings.pso_settings.initial_ub : BMO_MATOPS_ARRAY_ADD_SCALAR(init_out_vals,  0.5);

    sampling_bounds_check(vals_bound, n_vals, bounds_type, lower_bounds, upper_bounds, par_initial_lb, par_initial_ub);

    // parallelization setup

    int omp_n_threads = 1;

#ifdef OPTIM_USE_OPENMP
    if (settings.pso_settings.omp_n_threads > 0) {
        omp_n_threads = settings.pso_settings.omp_n_threads;
    } else {
        omp_n_threads = std::max(1, static_cast<int>(omp_get_max_threads()) / 2); // OpenMP often detects the number of virtual/logical cores, not physical cores
    }
#endif

    // random sampling setup

    rand_engine_t rand_engine(settings.rng_seed_value);
    std::vector<rand_engine_t> rand_engines_vec;

    for (int i = 0; i < omp_n_threads; ++i) {
        size_t seed_val = generate_seed_value(i, omp_n_threads, rand_engine);
        rand_engines_vec.push_back(rand_engine_t(seed_val));
    }

    // lambda function for box constraints

    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* box_data)> box_objfn \
    = [opt_objfn, vals_bound, bounds_type, lower_bounds, upper_bounds] (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data) \
    -> fp_t 
    {
        if (vals_bound) {
            ColVec_t vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);
            
            return opt_objfn(vals_inv_trans,nullptr,opt_data);
        }
        else
        {
            return opt_objfn(vals_inp,nullptr,opt_data);
        }
    };

    //
    // initialize

    ColVec_t rand_vec(n_vals);
    ColVec_t objfn_vals(n_pop);
    Mat_t P(n_pop, n_vals);

#ifdef OPTIM_USE_OPENMP
    #pragma omp parallel for num_threads(omp_n_threads) firstprivate(rand_vec)
#endif
    for (ompint_t i = 0; i < n_pop; ++i) {
        size_t thread_num = 0;

#ifdef OPTIM_USE_OPENMP
        thread_num = omp_get_thread_num();
#endif

        bmo::stats::internal::runif_vec_inplace<fp_t>(n_vals, rand_engines_vec[thread_num], rand_vec);

        P.row(i) = BMO_MATOPS_TRANSPOSE( par_initial_lb + BMO_MATOPS_HADAMARD_PROD( (par_initial_ub - par_initial_lb), rand_vec ) );

        fp_t prop_objfn_val = opt_objfn(BMO_MATOPS_TRANSPOSE(P.row(i)), nullptr, opt_data);

        if (std::isnan(prop_objfn_val)) {
            prop_objfn_val = inf;
        }
        
        objfn_vals(i) = prop_objfn_val;

        if (vals_bound) {
            P.row(i) = transform<RowVec_t>(P.row(i), bounds_type, lower_bounds, upper_bounds);
        }
    }

    ColVec_t best_vals = objfn_vals;

    Mat_t best_vecs = P;

    Mat_t V = BMO_MATOPS_ZERO_MAT(n_pop,n_vals);

    fp_t min_objfn_val_running = BMO_MATOPS_MIN_VAL(objfn_vals);
    fp_t min_objfn_val_check = min_objfn_val_running;

    RowVec_t best_sol_running = P.row( bmo::index_min(objfn_vals) );

    ColVec_t stag_vec = BMO_MATOPS_ZERO_COLVEC(n_pop); // arma::zeros(n_pop,1);

    //
    // begin loop

    size_t iter = 0;
    fp_t rel_objfn_change = 2.0*rel_objfn_change_tol;
    ColVec_t rand_CR(n_vals);

    while (rel_objfn_change > rel_objfn_change_tol && iter < n_gen) {
        ++iter;

        RowVec_t P_max = BMO_MATOPS_COLWISE_MAX(P);
        RowVec_t P_min = BMO_MATOPS_COLWISE_MIN(P);

#ifdef OPTIM_USE_OPENMP
        #pragma omp parallel for num_threads(omp_n_threads) firstprivate(rand_vec,rand_CR)
#endif
        for (ompint_t i = 0; i < n_pop; ++i) {
            size_t thread_num = 0;

#ifdef OPTIM_USE_OPENMP
            thread_num = omp_get_thread_num();
#endif

            uint_t c_1, c_2;

            do { // 'r_2' in paper's notation
                c_1 = bmo::stats::rind(0, n_pop-1, rand_engines_vec[thread_num]);
            } while(c_1 == i);

            do { // 'r_3' in paper's notation
                c_2 = bmo::stats::rind(0, n_pop-1, rand_engines_vec[thread_num]);
            } while(c_2 == i || c_2 == c_1);

            //

            bmo::stats::internal::runif_vec_inplace<fp_t>(n_vals, rand_engines_vec[thread_num], rand_CR);

            RowVec_t delta_vec = P.row(c_1) - P.row(c_2);

            for (size_t k = 0; k < n_vals; ++k) {
                if (rand_CR(k) <= par_CR) {
                    fp_t rand_u = bmo::stats::runif<fp_t>(rand_engines_vec[thread_num]);

                    V(i,k) = par_w * V(i,k) + par_beta * delta_vec(k) + par_c_2 * rand_u * (best_sol_running(k) - P(i,k));
                }
            }

            RowVec_t TR = P.row(i) + V.row(i);
            fp_t TR_objfn_val = box_objfn( BMO_MATOPS_TRANSPOSE(TR), nullptr, opt_data);

            if (TR_objfn_val < objfn_vals(i)) {
                P.row(i) = TR;
                objfn_vals(i) = TR_objfn_val;
            } else {
                stag_vec(i) += 1;
            }

            if (stag_vec(i) >= stag_limit) {
                bmo::stats::internal::runif_vec_inplace<fp_t>(n_vals, rand_engines_vec[thread_num], rand_vec);

                P.row(i) = P_min + BMO_MATOPS_HADAMARD_PROD(rand_vec, (P_max - P_min));
                stag_vec(i) = 0;

                objfn_vals(i) = box_objfn( BMO_MATOPS_TRANSPOSE(P.row(i)), nullptr, opt_data);
            }
                
            // if (objfn_vals(i) < best_vals(i)) {
            //     best_vals(i) = objfn_vals(i);
            //     best_vecs.row(i) = P.row(i);
            // }
        }

        par_w *= par_damp;
        // par_w = std::min(0.4,par_w*par_damp);

        //

        size_t min_objfn_val_index = bmo::index_min(objfn_vals);
        fp_t min_objfn_val = objfn_vals(min_objfn_val_index);

        if (min_objfn_val < min_objfn_val_running) {
            min_objfn_val_running = min_objfn_val;
            best_sol_running = P.row( min_objfn_val_index );
        }

        if (iter % check_freq == 0) {
            rel_objfn_change = std::abs(min_objfn_val_running - min_objfn_val_check) / (OPTIM_FPN_SMALL_NUMBER + std::abs(min_objfn_val_running));
            
            if (min_objfn_val_running < min_objfn_val_check) {
                min_objfn_val_check = min_objfn_val_running;
            }
        }

        //

        OPTIM_PSODV_TRACE(iter, rel_objfn_change, min_objfn_val_running, min_objfn_val_check, best_sol_running, par_w, P);
    }

    //

    if (vals_bound) {
        best_sol_running = inv_transform(best_sol_running, bounds_type, lower_bounds, upper_bounds);
    }

    error_reporting(init_out_vals, BMO_MATOPS_TRANSPOSE(best_sol_running), opt_objfn, opt_data, 
                    success, rel_objfn_change, rel_objfn_change_tol, iter, n_gen, 
                    conv_failure_switch, settings_inp);

    //
    
    return success;
}

optimlib_inline
bool
optim::pso_dv(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data
)
{
    return internal::pso_dv_impl(init_out_vals,opt_objfn,opt_data,nullptr);
}

optimlib_inline
bool
optim::pso_dv(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t& settings
)
{
    return internal::pso_dv_impl(init_out_vals,opt_objfn,opt_data,&settings);
}
