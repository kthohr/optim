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
 * Sequential unconstrained minimization technique (SUMT)
 */

#ifndef _optim_sumt_HPP
#define _optim_sumt_HPP

bool sumt_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data,
              std::function<arma::vec (const arma::vec& vals_inp, arma::mat* jacob_out, void* constr_data)> constr_fn, void* constr_data, algo_settings_t* settings_inp);

bool sumt(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data,
          std::function<arma::vec (const arma::vec& vals_inp, arma::mat* jacob_out, void* constr_data)> constr_fn, void* constr_data);

bool sumt(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data,
          std::function<arma::vec (const arma::vec& vals_inp, arma::mat* jacob_out, void* constr_data)> constr_fn, void* constr_data, algo_settings_t& settings);

struct sumt_struct {
    double c_pen;
};

#endif
