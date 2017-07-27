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
 * Differential Evolution (DE) optimization
 *
 * Keith O'Hara
 * 12/19/2016
 *
 * This version:
 * 07/19/2017
 */

#include "optim.hpp"

bool
optim::de_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data, double* value_out, optim_opt_settings* opt_params)
{
    bool success = false;

    const double BIG_POS_VAL = OPTIM_BIG_POS_NUM;

    const int conv_failure_switch = (opt_params) ? opt_params->conv_failure_switch : OPTIM_CONV_FAILURE_POLICY;
    const int n_gen = (opt_params) ? opt_params->de_n_gen : OPTIM_DEFAULT_DE_NGEN;
    const double err_tol = (opt_params) ? opt_params->err_tol : OPTIM_DEFAULT_ERR_TOL;

    const double par_F  = (opt_params) ? opt_params->de_F  : OPTIM_DEFAULT_DE_F; // tuning parameters
    const double par_CR = (opt_params) ? opt_params->de_CR : OPTIM_DEFAULT_DE_CR;
    //
    const int n_vals = init_out_vals.n_elem;
    const int N = n_vals*10;

    const int check_int = (n_gen >= 500) ? 100 : 20;

    int c_1, c_2, c_3, j, k;
    //
    double prop_objfn_val = 0.0;
    arma::vec X_prop = init_out_vals, past_objfn_vals(N), rand_unif(n_vals);
    arma::mat X(N,n_vals);

    for (int i=0; i < N; i++) {
        X.row(i) = init_out_vals.t() + arma::randu(1,n_vals) - 0.5;

        prop_objfn_val = opt_objfn(X.row(i).t(),nullptr,opt_data);

        if (std::isnan(prop_objfn_val)) {
            prop_objfn_val = BIG_POS_VAL;
        }
        
        past_objfn_vals(i) = prop_objfn_val;
    }

    double best_objfn_val = past_objfn_vals.min();
    //
    int iter = 0;
    double err = 2*err_tol;

    while (err > err_tol && iter < n_gen) {
        iter++;
        //
        for (int i=0; i < N; i++) {
            do { // 'a' in paper's notation
                c_1 = arma::as_scalar(arma::randi(1, arma::distr_param(0, N-1)));
            } while(c_1==i);

            do { // 'b' in paper's notation
                c_2 = arma::as_scalar(arma::randi(1, arma::distr_param(0, N-1)));
            } while(c_2==i || c_2==c_1);

            do { // 'c' in paper's notation
                c_3 = arma::as_scalar(arma::randi(1, arma::distr_param(0, N-1)));
            } while(c_3==i || c_3==c_1 || c_3==c_2);

            //
            j = arma::as_scalar(arma::randi(1, arma::distr_param(0, n_vals-1)));

            rand_unif = arma::randu(n_vals);
            k = 0;
            do {
                X_prop(j) = X(c_3,j) + par_F*(X(c_1,j) - X(c_2,j));

                j = (j+1)%n_vals;
                k++;
            } while((k < n_vals) && (rand_unif(k) < par_CR));
            //
            prop_objfn_val = opt_objfn(X_prop,nullptr,opt_data);
            
            if (prop_objfn_val <= past_objfn_vals(i)) {
                X.row(i) = X_prop.t();
                past_objfn_vals(i) = prop_objfn_val;
            } else {
                X_prop = X.row(i).t();
            }
        }
        //
        if (iter%check_int == 0) {
            err = std::abs(past_objfn_vals.min() - best_objfn_val);
            best_objfn_val = past_objfn_vals.min();
        }
    }
    //
    arma::uword min_i = past_objfn_vals.index_min();
    X_prop = X.row(min_i).t();

    error_reporting(init_out_vals,X_prop,opt_objfn,opt_data,success,value_out,err,err_tol,iter,n_gen,conv_failure_switch);
    //
    return success;
}

bool
optim::de(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data)
{
    return de_int(init_out_vals,opt_objfn,opt_data,nullptr,nullptr);
}

bool
optim::de(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data, optim_opt_settings& opt_params)
{
    return de_int(init_out_vals,opt_objfn,opt_data,nullptr,&opt_params);
}

bool
optim::de(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data, double& value_out)
{
    return de_int(init_out_vals,opt_objfn,opt_data,&value_out,nullptr);
}

bool
optim::de(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data, double& value_out, optim_opt_settings& opt_params)
{
    return de_int(init_out_vals,opt_objfn,opt_data,&value_out,&opt_params);
}
