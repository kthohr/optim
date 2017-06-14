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
 * Generic input to constrained optimization routines
 *
 * Keith O'Hara
 * 01/11/2017
 *
 * This version:
 * 06/12/2017
 */

#ifndef _optim_generic_constr_optim_HPP
#define _optim_generic_constr_optim_HPP

// without box constraints

bool generic_constr_optim_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                              std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
                              double* value_out, optim_opt_settings* opt_params);

bool generic_constr_optim(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data);

bool generic_constr_optim(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
                          optim_opt_settings& opt_params);

bool generic_constr_optim(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
                          double& value_out);

bool generic_constr_optim(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
                          double& value_out, optim_opt_settings& opt_params);

// with box constraints

bool generic_constr_optim_int(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
					          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                              std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
                              double* value_out, optim_opt_settings* opt_params);

bool generic_constr_optim(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
						  std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data);

bool generic_constr_optim(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
						  std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
                          optim_opt_settings& opt_params);

bool generic_constr_optim(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
						  std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
                          double& value_out);

bool generic_constr_optim(arma::vec& init_out_vals, const arma::vec& lower_bounds, const arma::vec& upper_bounds, 
						  std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
                          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
                          double& value_out, optim_opt_settings& opt_params);

#endif
