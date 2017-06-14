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
 * Sequential unconstrained minimization technique (SUMT)
 *
 * Keith O'Hara
 * 01/15/2016
 *
 * This version:
 * 06/11/2017
 */

#ifndef _optim_sumt_HPP
#define _optim_sumt_HPP

bool sumt_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
              std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
              double* value_out, optim_opt_settings* opt_params);

bool sumt(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data);

bool sumt(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
          optim_opt_settings& opt_params);

bool sumt(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
          double& value_out);

bool sumt(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data,
          std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* constr_data)> constr_fn, void* constr_data,
          double& value_out, optim_opt_settings& opt_params);

struct sumt_struct {
    double c_pen;
};

#endif
