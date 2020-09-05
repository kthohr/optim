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
 * Optimization control parameters
 */

#ifndef optim_structs_HPP
#define optim_structs_HPP

// BFGS

struct bfgs_settings_t
{
    double wolfe_cons_1 = 1E-03; // line search tuning parameter
    double wolfe_cons_2 = 0.90;  // line search tuning parameter
};

// Conjugate Gradient

struct cg_settings_t
{
    bool use_rel_sol_change_crit = false;
    int method = 2;
    double restart_threshold = 0.1;

    double wolfe_cons_1 = 1E-03; // line search tuning parameter
    double wolfe_cons_2 = 0.10;  // line search tuning parameter
};

// Gradient Descent

struct gd_settings_t
{
    int method = 0;

    // step size, or 'the learning rate'
    double par_step_size = 0.1;

    // decay
    bool step_decay = false;

    uint_t step_decay_periods = 10;
    double step_decay_val = 0.5;

    // momentum parameter
    double par_momentum = 0.9;

    // Ada parameters
    double par_ada_norm_term = 1.0e-08;

    double par_ada_rho = 0.9;

    bool ada_max = false;

    // Adam parameters
    double par_adam_beta_1 = 0.9;
    double par_adam_beta_2 = 0.999;

    // gradient clipping settings
    bool clip_grad = false;
    
    bool clip_max_norm = false;
    bool clip_min_norm = false;
    int clip_norm_type = 2;
    double clip_norm_bound = 5.0;
};

// L-BFGS

struct lbfgs_settings_t
{
    size_t par_M = 10;

    double wolfe_cons_1 = 1E-03; // line search tuning parameter
    double wolfe_cons_2 = 0.90;  // line search tuning parameter
};

// Nelder-Mead

struct nm_settings_t
{
    bool adaptive_pars = true;

    double par_alpha = 1.0; // reflection parameter
    double par_beta  = 0.5; // contraction parameter
    double par_gamma = 2.0; // expansion parameter
    double par_delta = 0.5; // shrinkage parameter
};

// DE

struct de_settings_t
{
    size_t n_pop = 200;
    size_t n_pop_best = 6;
    size_t n_gen = 1000;

    int mutation_method = 1; // 1 = rand; 2 = best

    size_t check_freq = (size_t)-1;

    double par_F = 0.8;
    double par_CR = 0.9;
    
    // DE-PRMM specific

    int pmax = 4;
    size_t max_fn_eval = 100000;

    double par_F_l = 0.1;
    double par_F_u = 1.0;

    double par_tau_F  = 0.1;
    double par_tau_CR = 0.1;

    double par_d_eps = 0.5;

    // initial vals

    Vec_t initial_lb; // this will default to -0.5
    Vec_t initial_ub; // this will default to  0.5

    //

    bool return_population_mat = false;
    Mat_t population_mat; // n_pop x n_vals
};

// PSO

struct pso_settings_t
{
    bool center_particle = true;

    size_t n_pop = 100;
    size_t n_gen = 1000;

    int inertia_method = 1; // 1 for linear decreasing between w_min and w_max; 2 for dampening

    size_t check_freq = (size_t)-1;

    double par_initial_w = 1.0;
    double par_w_damp = 0.99;

    double par_w_min = 0.10;
    double par_w_max = 0.99;

    int velocity_method = 1; // 1 for fixed; 2 for linear

    double par_c_cog = 2.0;
    double par_c_soc = 2.0;

    double par_initial_c_cog = 2.5;
    double par_final_c_cog   = 0.5;
    double par_initial_c_soc = 0.5;
    double par_final_c_soc   = 2.5;

    Vec_t initial_lb; // this will default to -0.5
    Vec_t initial_ub; // this will default to  0.5

    //

    bool return_position_mat = false;
    Mat_t position_mat; // n_pop x n_vals
};

// SUMT

struct sumt_settings_t
{
    double par_eta = 10.0;
};

// Broyden

struct broyden_settings_t
{
    double par_rho = 0.9;
    double par_sigma_1 = 0.001;
    double par_sigma_2 = 0.001;
};

// 

struct algo_settings_t
{
    // print and convergence options

    int print_level = 0;
    int conv_failure_switch = 0;

    // error tolerance and maxiumum iterations

    size_t iter_max = 2000;

    double grad_err_tol  = 1E-08;
    double rel_sol_change_tol  = 1E-14;
    double rel_objfn_change_tol = 1E-08;

    // bounds

    bool vals_bound = false;
    
    Vec_t lower_bounds;
    Vec_t upper_bounds;

    // values returned upon successful completion

    double opt_fn_value;      // will be returned by the optimization algorithm
    Vec_t opt_root_fn_values; // will be returned by the root-finding method

    size_t opt_iter;
    double opt_error_value;

    // algorithm-specific parameters

    // BFGS
    bfgs_settings_t bfgs_settings;

    // CG
    cg_settings_t cg_settings;

    // DE
    de_settings_t de_settings;

    // GD
    gd_settings_t gd_settings;

    // L-BFGS
    lbfgs_settings_t lbfgs_settings;

    // Nelder-Mead
    nm_settings_t nm_settings;

    // PSO
    pso_settings_t pso_settings;

    // SUMT
    sumt_settings_t sumt_settings;

    // Broyden
    broyden_settings_t broyden_settings;
};

#endif
