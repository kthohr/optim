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
 * Generic input to optimization routines
 *
 * Keith O'Hara
 * 01/11/2017
 *
 * This version:
 * 07/19/2017
 */

#ifndef _optim_generic_optim_HPP
#define _optim_generic_optim_HPP

// without box constraints

bool generic_optim_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data, 
                       double* value_out, optim_opt_settings* opt_params);

bool generic_optim(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data);

bool generic_optim(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                   optim_opt_settings& opt_params);

bool generic_optim(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                   double& value_out);

bool generic_optim(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                   double& value_out, optim_opt_settings& opt_params);

// with box constraints

bool generic_optim_int(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
                       std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                       double* value_out, optim_opt_settings* opt_params);

bool generic_optim(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
                   std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data);

bool generic_optim(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
                   std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                   optim_opt_settings& opt_params);

bool generic_optim(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
                   std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                   double& value_out);

bool generic_optim(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
                   std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                   double& value_out, optim_opt_settings& opt_params);

// jacobian adjustment for box constraints
arma::mat jacob_matrix_logit(const arma::vec& trans_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds);

#endif
