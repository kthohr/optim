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

#ifndef optim_structs_HPP
#define optim_structs_HPP

struct optim_opt_settings {
    // general
    int conv_failure_switch = OPTIM_CONV_FAILURE_POLICY;
    int iter_max = OPTIM_DEFAULT_ITER_MAX;
    double err_tol = OPTIM_DEFAULT_ERR_TOL;

    // SUMT parameter
    double sumt_par_eta = OPTIM_DEFAULT_SUMT_PENALTY_GROWTH;

    // CG
    int cg_method = OPTIM_DEFAULT_CG_METHOD;
    double cg_restart_threshold = OPTIM_DEFAULT_CG_RESTART_THRESHOLD;

    // DE
    int de_n_pop = -1;
    int de_n_pop_best = -1;
    int de_n_gen = -1;
    int de_pmax = -1;
    int de_max_fn_eval = -1;

    int de_mutation_method = 1; // 1 = rand; 2 = best

    int de_check_freq = -1;

    double de_par_F = OPTIM_DEFAULT_DE_PAR_F;
    double de_par_CR = OPTIM_DEFAULT_DE_PAR_CR;

    double de_par_F_l = -1;
    double de_par_F_u = -1;

    double de_par_tau_F  = -1;
    double de_par_tau_CR = -1;

    arma::vec de_lb; // this will default to -0.5
    arma::vec de_ub; // this will default to  0.5

    // Nelder-Mead
    double nm_par_alpha = OPTIM_DEFAULT_NM_PAR_ALPHA;
    double nm_par_beta  = OPTIM_DEFAULT_NM_PAR_BETA;
    double nm_par_gamma = OPTIM_DEFAULT_NM_PAR_GAMMA;
    double nm_par_delta = OPTIM_DEFAULT_NM_PAR_DELTA;

    // PSO
    int pso_n_pop = -1;
    int pso_n_gen = -1;

    arma::vec pso_lb; // this will default to -0.5
    arma::vec pso_ub; // this will default to  0.5
};

#endif
