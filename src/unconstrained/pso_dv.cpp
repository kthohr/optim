/*################################################################################
  ##
  ##   Copyright (C) 2016-2020 Keith O'Hara
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
optim::pso_dv_int(Vec_t& init_out_vals, 
                  std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
                  void* opt_data, 
                  algo_settings_t* settings_inp)
{
    bool success = false;

    const size_t n_vals = OPTIM_MATOPS_SIZE(init_out_vals);

    //
    // PSO settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const double err_tol = settings.err_tol;

    const size_t n_pop = (settings.pso_n_pop > 0) ? settings.pso_n_pop : 100;
    const size_t n_gen = (settings.pso_n_gen > 0) ? settings.pso_n_gen : 1000;

    const uint_t stag_limit = 50;

    double par_w = 1.0;
    double par_beta = 0.5;
    const double par_damp = 0.99;
    // const double par_c_1 = 1.494;
    const double par_c_2 = 1.494;

    const double par_CR = 0.7;

    const Vec_t par_initial_lb = ( OPTIM_MATOPS_SIZE(settings.pso_initial_lb) == n_vals ) ? settings.pso_initial_lb : OPTIM_MATOPS_ARRAY_ADD_SCALAR(init_out_vals, -0.5);
    const Vec_t par_initial_ub = ( OPTIM_MATOPS_SIZE(settings.pso_initial_ub) == n_vals ) ? settings.pso_initial_ub : OPTIM_MATOPS_ARRAY_ADD_SCALAR(init_out_vals,  0.5);

    const bool vals_bound = settings.vals_bound;
    
    const Vec_t lower_bounds = settings.lower_bounds;
    const Vec_t upper_bounds = settings.upper_bounds;

    const VecInt_t bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    //
    // lambda function for box constraints

    std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* box_data)> box_objfn \
    = [opt_objfn, vals_bound, bounds_type, lower_bounds, upper_bounds] (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data) \
    -> double 
    {
        if (vals_bound) {
            Vec_t vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);
            
            return opt_objfn(vals_inv_trans,nullptr,opt_data);
        }
        else
        {
            return opt_objfn(vals_inp,nullptr,opt_data);
        }
    };

    //
    // initialize

    Vec_t objfn_vals(n_pop);
    Mat_t P(n_pop,n_vals);

#ifdef OPTIM_USE_OMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < n_pop; ++i) {
        P.row(i) = OPTIM_MATOPS_TRANSPOSE( OPTIM_MATOPS_HADAMARD_PROD( (par_initial_lb + (par_initial_ub - par_initial_lb)), OPTIM_MATOPS_RANDU_VEC(n_vals) ) ); // arma::randu(1,n_vals)

        double prop_objfn_val = opt_objfn(OPTIM_MATOPS_TRANSPOSE(P.row(i)), nullptr, opt_data);

        if (std::isnan(prop_objfn_val)) {
            prop_objfn_val = inf;
        }
        
        objfn_vals(i) = prop_objfn_val;

        if (vals_bound) {
            P.row(i) = OPTIM_MATOPS_TRANSPOSE( transform(OPTIM_MATOPS_TRANSPOSE(P.row(i)), bounds_type, lower_bounds, upper_bounds) );
        }
    }

    Vec_t best_vals = objfn_vals;

    Mat_t best_vecs = P;

    Mat_t V = OPTIM_MATOPS_ZERO_MAT(n_pop,n_vals);

    double global_best_val = OPTIM_MATOPS_MIN_VAL(objfn_vals);
    RowVec_t global_best_vec = P.row( index_min(objfn_vals) );

    Vec_t stag_vec = OPTIM_MATOPS_ZERO_VEC(n_pop); // arma::zeros(n_pop,1);

    //
    // begin loop

    uint_t iter = 0;
    double err = 2.0*err_tol;

    while (err > err_tol && iter < n_gen) {
        iter++;

        RowVec_t P_max = OPTIM_MATOPS_COLWISE_MAX(P);
        RowVec_t P_min = OPTIM_MATOPS_COLWISE_MIN(P);

#ifdef OPTIM_USE_OMP
        #pragma omp parallel for 
#endif
        for (size_t i = 0; i < n_pop; ++i) {
            uint_t c_1, c_2;

            do { // 'r_2' in paper's notation
                c_1 = OPTIM_MATOPS_AS_SCALAR( OPTIM_MATOPS_RANDI_VEC(1, 0, n_pop-1) ); // arma::as_scalar(arma::randi(1, arma::distr_param(0, n_pop-1)));
            } while(c_1 == i);

            do { // 'r_3' in paper's notation
                c_2 = OPTIM_MATOPS_AS_SCALAR( OPTIM_MATOPS_RANDI_VEC(1, 0, n_pop-1) ); // arma::as_scalar(arma::randi(1, arma::distr_param(0, n_pop-1)));
            } while(c_2 == i || c_2 == c_1);

            //

            Vec_t rand_CR = OPTIM_MATOPS_RANDU_VEC(n_vals);

            RowVec_t delta_vec = P.row(c_1) - P.row(c_2);

            for (size_t k = 0; k < n_vals; ++k) {
                if (rand_CR(k) <= par_CR) {
                    double rand_u = OPTIM_MATOPS_AS_SCALAR( OPTIM_MATOPS_RANDU_VEC(1) );

                    V(i,k) = par_w*V(i,k) + par_beta*delta_vec(k) + par_c_2*rand_u*(global_best_vec(k) - P(i,k));
                }
            }

            RowVec_t TR = P.row(i) + V.row(i);
            double TR_objfn_val = box_objfn( OPTIM_MATOPS_TRANSPOSE(TR), nullptr, opt_data);

            if (TR_objfn_val < objfn_vals(i)) {
                P.row(i) = TR;
                objfn_vals(i) = TR_objfn_val;
            } else {
                stag_vec(i) += 1;
            }

            if (stag_vec(i) >= stag_limit) {
                P.row(i) = P_min + OPTIM_MATOPS_HADAMARD_PROD( OPTIM_MATOPS_RANDU_ROWVEC(n_vals), (P_max - P_min));
                stag_vec(i) = 0;

                objfn_vals(i) = box_objfn( OPTIM_MATOPS_TRANSPOSE(P.row(i)), nullptr, opt_data);
            }
                
            // if (objfn_vals(i) < best_vals(i)) {
            //     best_vals(i) = objfn_vals(i);
            //     best_vecs.row(i) = P.row(i);
            // }
        }

        size_t min_objfn_val_index = index_min(objfn_vals);
        double min_objfn_val = objfn_vals(min_objfn_val_index);

        if (min_objfn_val < global_best_val) {
            global_best_val = min_objfn_val;
            global_best_vec = P.row( min_objfn_val_index );
        }

        par_w *= par_damp;
        // par_w = std::min(0.4,par_w*par_damp);
    }

    //

    if (vals_bound) {
        global_best_vec = OPTIM_MATOPS_TRANSPOSE( inv_transform( OPTIM_MATOPS_TRANSPOSE(global_best_vec), bounds_type, lower_bounds, upper_bounds) );
    }

    error_reporting(init_out_vals, OPTIM_MATOPS_TRANSPOSE(global_best_vec), opt_objfn, opt_data, success, err, err_tol, iter, n_gen, conv_failure_switch, settings_inp);

    //
    
    return true;
}

optimlib_inline
bool
optim::pso_dv(Vec_t& init_out_vals, 
              std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
              void* opt_data)
{
    return pso_dv_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

optimlib_inline
bool
optim::pso_dv(Vec_t& init_out_vals, 
              std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
              void* opt_data, 
              algo_settings_t& settings)
{
    return pso_dv_int(init_out_vals,opt_objfn,opt_data,&settings);
}
