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
 * Nelder-Mead
 */

#include "optim.hpp"

// [OPTIM_BEGIN]
optimlib_inline
bool
optim::internal::nm_impl(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t* settings_inp
)
{
    bool success = false;

    //
    // NM settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const size_t n_vals = (settings.nm_settings.custom_initial_simplex) ? BMO_MATOPS_NCOL(settings.nm_settings.initial_simplex_points) : BMO_MATOPS_SIZE(init_out_vals);

    const int print_level = settings.print_level;
    
    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const size_t iter_max = settings.iter_max;
    const fp_t rel_objfn_change_tol = settings.rel_objfn_change_tol;
    const fp_t rel_sol_change_tol = settings.rel_sol_change_tol;

    // expansion / contraction parameters
    
    const fp_t par_alpha = settings.nm_settings.par_alpha;
    const fp_t par_beta  = (settings.nm_settings.adaptive_pars) ? 0.75 - 1.0 / (2.0*n_vals) : settings.nm_settings.par_beta;
    const fp_t par_gamma = (settings.nm_settings.adaptive_pars) ? 1.0 + 2.0 / n_vals        : settings.nm_settings.par_gamma;
    const fp_t par_delta = (settings.nm_settings.adaptive_pars) ? 1.0 - 1.0 / n_vals        : settings.nm_settings.par_delta;

    const bool vals_bound = settings.vals_bound;
    
    const ColVec_t lower_bounds = settings.lower_bounds;
    const ColVec_t upper_bounds = settings.upper_bounds;

    const ColVecInt_t bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    // random sampling setup

    int omp_n_threads = 1;

#ifdef OPTIM_USE_OMP
    if (settings.nm_settings.omp_n_threads > 0) {
        omp_n_threads = settings.nm_settings.omp_n_threads;
    } else {
        omp_n_threads = std::max(1, static_cast<int>(omp_get_max_threads()) / 2); // OpenMP often detects the number of virtual/logical cores, not physical cores
    }
#else
    (void)(omp_n_threads);
#endif

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
    // setup

    ColVec_t simplex_fn_vals(n_vals+1);
    ColVec_t simplex_fn_vals_old(n_vals+1);
    Mat_t simplex_points(n_vals+1, n_vals);
    Mat_t simplex_points_old(n_vals+1, n_vals);

    if (settings.nm_settings.custom_initial_simplex) {
        simplex_points = settings.nm_settings.initial_simplex_points;
        simplex_fn_vals(0) = opt_objfn(BMO_MATOPS_TRANSPOSE(simplex_points.row(0)), nullptr, opt_data);
    } else {
        simplex_fn_vals(0) = opt_objfn(init_out_vals, nullptr, opt_data);
        simplex_points.row(0) = BMO_MATOPS_TRANSPOSE(init_out_vals);
    }

    if (vals_bound) {
        simplex_points.row(0) = transform<RowVec_t>(simplex_points.row(0), bounds_type, lower_bounds, upper_bounds);
    }

    for (size_t i = 1; i < n_vals + 1; ++i) {
        if (!settings.nm_settings.custom_initial_simplex) {
            if (init_out_vals(i-1) != 0.0) {
                simplex_points.row(i) = BMO_MATOPS_TRANSPOSE( init_out_vals + 0.05*init_out_vals(i-1) * unit_vec(i-1,n_vals) );
            } else {
                simplex_points.row(i) = BMO_MATOPS_TRANSPOSE( init_out_vals + 0.00025 * unit_vec(i-1,n_vals) );
                // simplex_points.row(i) = init_out_vals.t() + 0.05*arma::trans(unit_vec(i-1,n_vals));
            }
        }

        simplex_fn_vals(i) = opt_objfn(BMO_MATOPS_TRANSPOSE(simplex_points.row(i)),nullptr,opt_data);

        if (vals_bound) {
            simplex_points.row(i) = transform<RowVec_t>(simplex_points.row(i), bounds_type, lower_bounds, upper_bounds);
        }
    }

    fp_t min_val = BMO_MATOPS_MIN_VAL(simplex_fn_vals);

    //
    // begin loop

    if (print_level > 0) {
        std::cout << "\nNelder-Mead: beginning search...\n";

        if (print_level >= 3) {
            std::cout << "  - Initialization Phase:\n";
            std::cout << "    Objective function value at each vertex:\n";
            BMO_MATOPS_COUT << BMO_MATOPS_TRANSPOSE(simplex_fn_vals) << "\n";
            std::cout << "    Simplex matrix:\n"; 
            BMO_MATOPS_COUT << simplex_points << "\n";
        }
    }

    size_t iter = 0;
    fp_t rel_objfn_change = 2*std::abs(rel_objfn_change_tol);
    fp_t rel_sol_change = 2*std::abs(rel_sol_change_tol);

    simplex_fn_vals_old = simplex_fn_vals;
    simplex_points_old = simplex_points;

    while (rel_objfn_change > rel_objfn_change_tol && rel_sol_change > rel_sol_change_tol && iter < iter_max) {
        ++iter;
        bool next_iter = false;
        
        // step 1

        // ColVecInt_t sort_vec = arma::sort_index(simplex_fn_vals); // sort from low (best) to high (worst) values
        ColVecUInt_t sort_vec = get_sort_index(simplex_fn_vals); // sort from low (best) to high (worst) values

        simplex_fn_vals = BMO_MATOPS_EVAL(simplex_fn_vals(sort_vec));
        simplex_points = BMO_MATOPS_EVAL(BMO_MATOPS_ROWS(simplex_points, sort_vec));

        // step 2

        ColVec_t centroid = BMO_MATOPS_TRANSPOSE( BMO_MATOPS_COLWISE_SUM( BMO_MATOPS_MIDDLE_ROWS(simplex_points, 0, n_vals-1) ) ) / static_cast<fp_t>(n_vals);

        ColVec_t x_r = centroid + par_alpha*(centroid - BMO_MATOPS_TRANSPOSE(simplex_points.row(n_vals)));

        fp_t f_r = box_objfn(x_r, nullptr, opt_data);

        if (f_r >= simplex_fn_vals(0) && f_r < simplex_fn_vals(n_vals-1)) {
            // reflected point is neither best nor worst in the new simplex
            simplex_points.row(n_vals) = BMO_MATOPS_TRANSPOSE(x_r);
            simplex_fn_vals(n_vals) = f_r;
            next_iter = true;
        }

        // step 3

        if (!next_iter && f_r < simplex_fn_vals(0)) {
            // reflected point is better than the current best; try to go farther along this direction
            ColVec_t x_e = centroid + par_gamma*(x_r - centroid);

            fp_t f_e = box_objfn(x_e, nullptr, opt_data);

            if (f_e < f_r) {
                simplex_points.row(n_vals) = BMO_MATOPS_TRANSPOSE(x_e);
                simplex_fn_vals(n_vals) = f_e;
            } else {
                simplex_points.row(n_vals) = BMO_MATOPS_TRANSPOSE(x_r);
                simplex_fn_vals(n_vals) = f_r;
            }

            next_iter = true;
        }

        // steps 4, 5, 6

        if (!next_iter && f_r >= simplex_fn_vals(n_vals-1)) {
            // reflected point is still worse than x_n; contract

            // steps 4 and 5

            if (f_r < simplex_fn_vals(n_vals)) {
                // outside contraction
                ColVec_t x_oc = centroid + par_beta*(x_r - centroid);

                fp_t f_oc = box_objfn(x_oc, nullptr, opt_data);

                if (f_oc <= f_r)
                {
                    simplex_points.row(n_vals) = BMO_MATOPS_TRANSPOSE(x_oc);
                    simplex_fn_vals(n_vals) = f_oc;
                    next_iter = true;
                }
            } else {
                // inside contraction: f_r >= simplex_fn_vals(n_vals)
                
                // x_ic = centroid - par_beta*(x_r - centroid);
                ColVec_t x_ic = centroid + par_beta*( BMO_MATOPS_TRANSPOSE(simplex_points.row(n_vals)) - centroid );

                fp_t f_ic = box_objfn(x_ic, nullptr, opt_data);

                if (f_ic < simplex_fn_vals(n_vals))
                {
                    simplex_points.row(n_vals) = BMO_MATOPS_TRANSPOSE(x_ic);
                    simplex_fn_vals(n_vals) = f_ic;
                    next_iter = true;
                }
            }
        }

        // step 6

        if (!next_iter) {
            // neither outside nor inside contraction was acceptable; shrink the simplex toward x(0)
            for (size_t i = 1; i < n_vals + 1; i++) {
                simplex_points.row(i) = simplex_points.row(0) + par_delta*(simplex_points.row(i) - simplex_points.row(0));
            }

#ifdef OPTIM_USE_OMP
            #pragma omp parallel for num_threads(omp_n_threads)
#endif
            for (size_t i = 1; i < n_vals + 1; i++) {
                simplex_fn_vals(i) = box_objfn( BMO_MATOPS_TRANSPOSE(simplex_points.row(i)), nullptr, opt_data);
            }
        }

        min_val = BMO_MATOPS_MIN_VAL(simplex_fn_vals);

        //

        // fp_t change_val_min = std::abs(min_val - BMO_MATOPS_MIN_VAL(simplex_fn_vals));
        // fp_t change_val_max = std::abs(min_val - BMO_MATOPS_MAX_VAL(simplex_fn_vals));
    
        // rel_objfn_change = std::max( change_val_min, change_val_max ) / (OPTIM_FPN_SMALL_NUMBER + BMO_MATOPS_ABS_MAX_VAL(simplex_fn_vals));

        rel_objfn_change = (BMO_MATOPS_ABS_MAX_VAL(simplex_fn_vals - simplex_fn_vals_old)) / (OPTIM_FPN_SMALL_NUMBER + BMO_MATOPS_ABS_MAX_VAL(simplex_fn_vals_old));
        simplex_fn_vals_old = simplex_fn_vals;

        if (rel_sol_change_tol >= 0.0) { 
            rel_sol_change = (BMO_MATOPS_ABS_MAX_VAL(simplex_points - simplex_points_old)) / (OPTIM_FPN_SMALL_NUMBER + BMO_MATOPS_ABS_MAX_VAL(simplex_points_old));
            simplex_points_old = simplex_points;
        }

        // printing

        OPTIM_NM_TRACE(iter, min_val, rel_objfn_change, rel_sol_change, simplex_fn_vals, simplex_points);
    }

    if (print_level > 0) {
        std::cout << "Nelder-Mead: search completed.\n";
    }

    //

    ColVec_t prop_out = BMO_MATOPS_TRANSPOSE(simplex_points.row(index_min(simplex_fn_vals)));
    
    if (vals_bound) {
        prop_out = inv_transform(prop_out, bounds_type, lower_bounds, upper_bounds);
    }

    error_reporting(init_out_vals, prop_out, opt_objfn, opt_data,
                    success, rel_objfn_change, rel_objfn_change_tol, iter, iter_max, 
                    conv_failure_switch, settings_inp);

    //
    
    return success;
}

optimlib_inline
bool
optim::nm(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data
)
{
    return internal::nm_impl(init_out_vals,opt_objfn,opt_data,nullptr);
}

optimlib_inline
bool
optim::nm(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t& settings
)
{
    return internal::nm_impl(init_out_vals,opt_objfn,opt_data,&settings);
}
