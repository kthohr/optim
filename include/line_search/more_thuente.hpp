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

/*
 * Moré and Thuente line search
 *
 * Based on MINPACK fortran code and Dianne P. O'Leary's Matlab translation of MINPACK
 *
 * Keith O'Hara
 * 01/03/2017
 *
 * This version:
 * 01/11/2017
 */

#ifndef _optim_more_thuente_HPP
#define _optim_more_thuente_HPP

double line_search_mt(double step, arma::vec& x, arma::vec& grad, const arma::vec& direc, double* wolfe_cons_1_inp, double* wolfe_cons_2_inp, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data);

// update the 'interval of uncertainty'
int mt_step(double& st_best, double& f_best, double& d_best, double& st_other, double& f_other, double& d_other, double& step, double& f_step, double& d_step, bool& bracket, double step_min, double step_max);
double mt_sup_norm(double a, double b, double c);

#endif
