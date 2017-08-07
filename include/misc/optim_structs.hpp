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

struct opt_settings {
    // general
    int conv_failure_switch = 0;
    int iter_max = 2000;
    double err_tol = 1E-08;

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

    double de_par_F = OPTIM_DEFAULT_DE_PAR_F;
    double de_par_CR = OPTIM_DEFAULT_DE_PAR_CR;

    double de_par_F_l = 0.1;
    double de_par_F_u = 1.0;

    double de_par_tau_F  = 0.1;
    double de_par_tau_CR = 0.1;

    arma::vec de_lb; // this will default to -0.5
    arma::vec de_ub; // this will default to  0.5

    // Nelder-Mead
    bool nm_adaptive= true;
    double nm_par_alpha = 1.0; // reflection parameter
    double nm_par_beta  = 0.5; // contraction parameter
    double nm_par_gamma = 2.0; // expansion parameter
    double nm_par_delta = 0.5; // shrinkage parameter

    // PSO
    int pso_n_pop = -1;
    int pso_n_gen = -1;

    arma::vec pso_lb; // this will default to -0.5
    arma::vec pso_ub; // this will default to  0.5
};

#endif
