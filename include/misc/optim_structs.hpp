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
 * Optimization control parameters
 */

#ifndef optim_structs_HPP
#define optim_structs_HPP

struct gd_settings_t
{
    // step size, or 'learning rate'
    double step_size = 0.1;

    // decay
    bool step_decay = false;

    uint_t step_decay_periods = 10;
    double step_decay_val = 0.5;

    // momentum parameter
    double momentum_par = 0.9;

    // Ada parameters
    double norm_term = 10e-08;

    double ada_rho = 0.9;

    bool ada_max = false;

    // Adam parameters
    double adam_beta_1 = 0.9;
    double adam_beta_2 = 0.999;
};

struct algo_settings_t
{
    // general
    int verbose_print_level = 0;
    int conv_failure_switch = 0;
    int iter_max = 2000;
    double err_tol = 1E-08;

    bool vals_bound = false;
    
    arma::vec lower_bounds;
    arma::vec upper_bounds;

    // returned by algorithms
    double opt_value;      // will be returned by the optimization algorithm
    arma::vec zero_values; // will be returned by the root-finding method

    int opt_iter;
    double opt_err;

    // SUMT parameter
    double sumt_par_eta = 10.0;

    // CG
    int cg_method = 2;
    double cg_restart_threshold = 0.1;

    // DE
    int de_n_pop = 200;
    int de_n_pop_best = 6;
    int de_n_gen = 1000;
    
    int de_pmax = 4;
    int de_max_fn_eval = 100000;

    int de_mutation_method = 1; // 1 = rand; 2 = best

    int de_check_freq = -1;

    double de_par_F = 0.8;
    double de_par_CR = 0.9;

    double de_par_F_l = 0.1;
    double de_par_F_u = 1.0;

    double de_par_tau_F  = 0.1;
    double de_par_tau_CR = 0.1;

    arma::vec de_initial_lb; // this will default to -0.5
    arma::vec de_initial_ub; // this will default to  0.5

    // GD
    int gd_method = 0;
    gd_settings_t gd_settings;

    // L-BFGS
    int lbfgs_par_M = 10;

    // Nelder-Mead
    bool nm_adaptive= true;

    double nm_par_alpha = 1.0; // reflection parameter
    double nm_par_beta  = 0.5; // contraction parameter
    double nm_par_gamma = 2.0; // expansion parameter
    double nm_par_delta = 0.5; // shrinkage parameter

    // PSO
    bool pso_center_particle = true;

    int pso_n_pop = 100;
    int pso_n_gen = 1000;

    int pso_inertia_method = 1; // 1 for linear decreasing between w_min and w_max; 2 for dampening

    int pso_check_freq = -1;

    double pso_par_initial_w = 1.0;
    double pso_par_w_damp = 0.99;

    double pso_par_w_min = 0.10;
    double pso_par_w_max = 0.99;

    int pso_velocity_method = 1; // 1 for fixed; 2 for linear

    double pso_par_c_cog = 2.0;
    double pso_par_c_soc = 2.0;

    double pso_par_initial_c_cog = 2.5;
    double pso_par_final_c_cog   = 0.5;
    double pso_par_initial_c_soc = 0.5;
    double pso_par_final_c_soc   = 2.5;

    arma::vec pso_initial_lb; // this will default to -0.5
    arma::vec pso_initial_ub; // this will default to  0.5
};

#endif
