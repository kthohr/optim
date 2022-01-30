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
 * Particle Swarm Optimization (PSO)
 */

#include "optim.hpp"

// [OPTIM_BEGIN]
optimlib_inline
bool
optim::internal::pso_impl(
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

    const bool center_particle = settings.pso_settings.center_particle;

    const size_t n_pop = (center_particle) ? settings.pso_settings.n_pop + 1 : settings.pso_settings.n_pop;
    const size_t n_gen = settings.pso_settings.n_gen;
    const size_t check_freq = settings.pso_settings.check_freq;

    const uint_t inertia_method = settings.pso_settings.inertia_method;

    fp_t par_w = settings.pso_settings.par_initial_w;
    const fp_t par_w_max = settings.pso_settings.par_w_max;
    const fp_t par_w_min = settings.pso_settings.par_w_min;
    const fp_t par_damp = settings.pso_settings.par_w_damp;

    const uint_t velocity_method = settings.pso_settings.velocity_method;

    fp_t par_c_cog = settings.pso_settings.par_c_cog;
    fp_t par_c_soc = settings.pso_settings.par_c_soc;

    const fp_t par_initial_c_cog = settings.pso_settings.par_initial_c_cog;
    const fp_t par_final_c_cog = settings.pso_settings.par_final_c_cog;
    const fp_t par_initial_c_soc = settings.pso_settings.par_initial_c_soc;
    const fp_t par_final_c_soc = settings.pso_settings.par_final_c_soc;

    const bool return_position_mat = settings.pso_settings.return_position_mat;

    const bool vals_bound = settings.vals_bound;
    
    const ColVec_t lower_bounds = settings.lower_bounds;
    const ColVec_t upper_bounds = settings.upper_bounds;

    const ColVecInt_t bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    ColVec_t par_initial_lb = ( BMO_MATOPS_SIZE(settings.pso_settings.initial_lb) == n_vals ) ? settings.pso_settings.initial_lb : BMO_MATOPS_ARRAY_ADD_SCALAR(init_out_vals, -0.5);
    ColVec_t par_initial_ub = ( BMO_MATOPS_SIZE(settings.pso_settings.initial_ub) == n_vals ) ? settings.pso_settings.initial_ub : BMO_MATOPS_ARRAY_ADD_SCALAR(init_out_vals,  0.5);

    sampling_bounds_check(vals_bound, n_vals, bounds_type, lower_bounds, upper_bounds, par_initial_lb, par_initial_ub);

    // random sampling setup

    int omp_n_threads = 1;
    rand_engine_t rand_engine(settings.rng_seed_value);
    std::vector<rand_engine_t> engines;

#ifdef OPTIM_USE_OMP
    if (settings.pso_settings.omp_n_threads > 0) {
        omp_n_threads = settings.pso_settings.omp_n_threads;
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
            
            return opt_objfn(vals_inv_trans,nullptr,opt_data);
        } else {
            return opt_objfn(vals_inp,nullptr,opt_data);
        }
    };

    //
    // initialize

    ColVec_t rand_vec(n_vals);
    ColVec_t objfn_vals(n_pop);
    Mat_t P(n_pop,n_vals);

#ifdef OPTIM_USE_OMP
    #pragma omp parallel for num_threads(omp_n_threads) private(rand_vec)
#endif
    for (size_t i = 0; i < n_pop; ++i) {
        size_t thread_num = 0;

#ifdef OPTIM_USE_OMP
        thread_num = omp_get_thread_num();
#endif

        if (center_particle && i == n_pop - 1) {
            P.row(i) = BMO_MATOPS_COLWISE_SUM( BMO_MATOPS_MIDDLE_ROWS(P, 0, n_pop-2) ) / static_cast<fp_t>(n_pop-1); // center vector
        } else {
            bmo_stats::internal::runif_vec_inplace<fp_t>(n_vals, engines[thread_num], rand_vec);

            P.row(i) = BMO_MATOPS_TRANSPOSE( par_initial_lb + BMO_MATOPS_HADAMARD_PROD( (par_initial_ub - par_initial_lb), rand_vec ) );
        }

        fp_t prop_objfn_val = opt_objfn(BMO_MATOPS_TRANSPOSE(P.row(i)), nullptr, opt_data);

        if (!std::isfinite(prop_objfn_val)) {
            prop_objfn_val = inf;
        }
        
        objfn_vals(i) = prop_objfn_val;

        if (vals_bound) {
            P.row(i) = transform<RowVec_t>(P.row(i), bounds_type, lower_bounds, upper_bounds);
        }
    }

    ColVec_t best_vals = objfn_vals;
    Mat_t best_vecs = P;

    fp_t min_objfn_val_running = BMO_MATOPS_MIN_VAL(objfn_vals);
    fp_t min_objfn_val_check = min_objfn_val_running;
    
    RowVec_t best_sol_running = P.row( index_min(objfn_vals) );

    //
    // begin loop

    size_t iter = 0;
    fp_t rel_objfn_change = 2.0*rel_objfn_change_tol;

    RowVec_t rand_vec_1(n_vals);
    RowVec_t rand_vec_2(n_vals);
    Mat_t V = BMO_MATOPS_ZERO_MAT(n_pop,n_vals);

    while (rel_objfn_change > rel_objfn_change_tol && iter < n_gen) {
        ++iter;
        
        //
        // parameter updating

        if (inertia_method == 1) {
            par_w = par_w_min + (par_w_max - par_w_min) * (iter + 1) / n_gen;
        } else {
            par_w *= par_damp;
        }

        if (velocity_method == 2) {
            par_c_cog = par_initial_c_cog - (par_initial_c_cog - par_final_c_cog) * (iter + 1) / n_gen;
            par_c_soc = par_initial_c_soc - (par_initial_c_soc - par_final_c_soc) * (iter + 1) / n_gen;
        }

        //
        // population loop

#ifdef OPTIM_USE_OMP
        #pragma omp parallel for num_threads(omp_n_threads) private(rand_vec_1,rand_vec_2)
#endif
        for (size_t i=0; i < n_pop; ++i) {
            size_t thread_num = 0;

#ifdef OPTIM_USE_OMP
            thread_num = omp_get_thread_num();
#endif

            if ( !(center_particle && i == n_pop - 1) ) {
                bmo_stats::internal::runif_vec_inplace<fp_t>(n_vals, engines[thread_num], rand_vec_1);
                bmo_stats::internal::runif_vec_inplace<fp_t>(n_vals, engines[thread_num], rand_vec_2);

                // RowVec_t rand_vec_1 = bmo_stats::runif_vec<fp_t, RowVec_t>(n_vals, engines[thread_num]);
                // RowVec_t rand_vec_2 = bmo_stats::runif_vec<fp_t, RowVec_t>(n_vals, engines[thread_num]);

                V.row(i) = par_w * V.row(i) + par_c_cog * BMO_MATOPS_HADAMARD_PROD( rand_vec_1, (best_vecs.row(i) - P.row(i)) ) \
                    + par_c_soc * BMO_MATOPS_HADAMARD_PROD( rand_vec_2, (best_sol_running - P.row(i)) );

                P.row(i) += V.row(i);
            } else {
                P.row(i) = BMO_MATOPS_COLWISE_SUM( BMO_MATOPS_MIDDLE_ROWS(P, 0, n_pop-2) ) / static_cast<fp_t>(n_pop-1); // center vector
            }
            
            //

            fp_t prop_objfn_val = box_objfn( BMO_MATOPS_TRANSPOSE(P.row(i)), nullptr, opt_data);

            if (!std::isfinite(prop_objfn_val)) {
                prop_objfn_val = inf;
            }
        
            objfn_vals(i) = prop_objfn_val;
                
            if (objfn_vals(i) < best_vals(i)) {
                best_vals(i) = objfn_vals(i);
                best_vecs.row(i) = P.row(i);
            }
        }

        size_t min_objfn_val_index = index_min(best_vals);
        fp_t min_objfn_val = best_vals(min_objfn_val_index);

        //

        if (min_objfn_val < min_objfn_val_running) {
            min_objfn_val_running = min_objfn_val;
            best_sol_running = best_vecs.row( min_objfn_val_index );
        }

        if (iter % check_freq == 0) {
            rel_objfn_change = std::abs(min_objfn_val_running - min_objfn_val_check) / (OPTIM_FPN_SMALL_NUMBER + std::abs(min_objfn_val_running));
            
            if (min_objfn_val_running < min_objfn_val_check) {
                min_objfn_val_check = min_objfn_val_running;
            }
        }

        //

        OPTIM_PSO_TRACE(iter, rel_objfn_change, min_objfn_val_running, min_objfn_val_check, best_sol_running, P);
    }

    //

    if (return_position_mat) {
        if (vals_bound) {
#ifdef OPTIM_USE_OMP
        #pragma omp parallel for num_threads(omp_n_threads)
#endif
            for (size_t i = 0; i < n_pop; ++i) {
                P.row(i) = inv_transform<RowVec_t>(P.row(i), bounds_type, lower_bounds, upper_bounds);
            }
        }

        settings_inp->pso_settings.position_mat = P;
    }

    //

    if (vals_bound) {
        best_sol_running = inv_transform( best_sol_running, bounds_type, lower_bounds, upper_bounds);
    }

    error_reporting(init_out_vals, BMO_MATOPS_TRANSPOSE(best_sol_running), opt_objfn, opt_data, 
                    success, rel_objfn_change, rel_objfn_change_tol, iter, n_gen, 
                    conv_failure_switch, settings_inp);

    //

    return true;
}

optimlib_inline
bool
optim::pso(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data
)
{
    return internal::pso_impl(init_out_vals,opt_objfn,opt_data,nullptr);
}

optimlib_inline
bool
optim::pso(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t& settings
)
{
    return internal::pso_impl(init_out_vals,opt_objfn,opt_data,&settings);
}
