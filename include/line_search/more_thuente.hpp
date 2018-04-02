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
 * Moré and Thuente line search
 *
 * Based on MINPACK fortran code and Dianne P. O'Leary's Matlab translation of MINPACK
 */

#ifndef _optim_more_thuente_HPP
#define _optim_more_thuente_HPP

double line_search_mt(double step, arma::vec& x, arma::vec& grad, const arma::vec& direc, const double* wolfe_cons_1_inp, const double* wolfe_cons_2_inp, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data);

// update the 'interval of uncertainty'
uint_t mt_step(double& st_best, double& f_best, double& d_best, double& st_other, double& f_other, double& d_other, double& step, double& f_step, double& d_step, bool& bracket, double step_min, double step_max);
double mt_sup_norm(const double a, const double b, const double c);

#endif
