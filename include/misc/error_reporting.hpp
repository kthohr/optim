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
 * Error reporting
 *
 * Keith O'Hara
 * 06/11/2016
 *
 * This version:
 * 06/12/2017
 */

#ifndef _optim_error_reporting_HPP
#define _optim_error_reporting_HPP

void error_reporting(arma::vec& out_vals, const arma::vec& x_p, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                     bool& success, double* value_out, double err, double err_tol, int iter, int iter_max, int conv_failure_switch);

void error_reporting(arma::vec& out_vals, const arma::vec& x_p, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                     bool success, double* value_out, int conv_failure_switch);

void error_reporting(arma::vec& out_vals, const arma::vec& x_p, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data,
                     bool& success, arma::vec* value_out, double err, double err_tol, int iter, int iter_max, int conv_failure_switch);

#include "error_reporting.ipp"

#endif
