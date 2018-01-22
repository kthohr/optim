/*################################################################################
  ##
  ##   Copyright (C) 2016-2018 Keith O'Hara
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
 * Newton's method for non-linear optimization
 */

#ifndef _optim_newton_HPP
#define _optim_newton_HPP

bool newton_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, arma::mat* hess_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings* settings_inp);

bool newton(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, arma::mat* hess_out, void* opt_data)> opt_objfn, void* opt_data);
bool newton(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, arma::mat* hess_out, void* opt_data)> opt_objfn, void* opt_data, algo_settings& settings);

#endif
