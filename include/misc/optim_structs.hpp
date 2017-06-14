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
    double eta = OPTIM_DEFAULT_PENALTY_GROWTH;

    // Nelder-Mead
    double alpha_nm = OPTIM_DEFAULT_NM_ALPHA;
    double beta_nm  = OPTIM_DEFAULT_NM_BETA;
    double gamma_nm = OPTIM_DEFAULT_NM_GAMMA;
    double delta_nm = OPTIM_DEFAULT_NM_DELTA;

    // CG
    int method_cg = OPTIM_DEFAULT_CG_METHOD;

    // DE
    int de_n_gen = OPTIM_DEFAULT_DE_NGEN;
    double de_F = OPTIM_DEFAULT_DE_F;
    double de_CR = OPTIM_DEFAULT_DE_CR;
};

#endif
