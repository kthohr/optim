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
 * Error reporting
 */

inline
void
error_reporting(arma::vec& out_vals, const arma::vec& x_p, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data,
                bool& success, const double err, const double err_tol, const int iter, const int iter_max, const int conv_failure_switch, algo_settings* settings_inp)
{
    success = false;

    if (conv_failure_switch == 0) {
        out_vals = x_p;

        if (err <= err_tol && iter <= iter_max) {
            success = true;
        }
    } else if (conv_failure_switch == 1) {
        out_vals = x_p;

        if (err <= err_tol && iter <= iter_max) {
            success = true;
        } else {
            printf("optim failure: iter_max reached before convergence could be achieved.\n");
            printf("optim failure: returned best guess.\n");
            
            std::cout << "iterations: " << iter << ". error: " << err << std::endl;
        }
    } else if (conv_failure_switch == 2) {
        if (err <= err_tol && iter <= iter_max) {
            out_vals = x_p;
            success = true;
        } else {
            printf("optim failure: iter_max reached before convergence could be achieved.\n");
            printf("optim failure: best guess:\n");

            arma::cout << x_p.t() << arma::endl;
            std::cout << "iterations: " << iter << ". error: " << err << std::endl;
        }
    } else {
        printf("optim failure: unrecognized conv_failure_switch value.\n");
        success = false;
    }
    //
    if (settings_inp) {
        settings_inp->opt_value = opt_objfn(x_p,nullptr,opt_data);
        settings_inp->opt_iter = iter;
        settings_inp->opt_err  = err;
    }
}

inline
void
error_reporting(arma::vec& out_vals, const arma::vec& x_p, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data,
                bool& success, const int conv_failure_switch, algo_settings* settings_inp)
{
    if (conv_failure_switch == 0 || conv_failure_switch == 1) {
        out_vals = x_p;
    } else if (conv_failure_switch == 2) {
        if (success) {
            out_vals = x_p;
        }
    } else {
        printf("optim failure: unrecognized conv_failure_switch value.\n");
        success = false;
    }
    //
    if (settings_inp) {
        settings_inp->opt_value = opt_objfn(x_p,nullptr,opt_data);
    }
}

inline
void
error_reporting(arma::vec& out_vals, const arma::vec& x_p, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data,
                bool& success, const double err, const double err_tol, const int iter, const int iter_max, const int conv_failure_switch, algo_settings* settings_inp)
{
    success = false;

    if (conv_failure_switch == 0) {
        out_vals = x_p;

        if (err <= err_tol && iter <= iter_max) {
            success = true;
        }
    } else if (conv_failure_switch == 1) {
        out_vals = x_p;

        if (err <= err_tol && iter <= iter_max) {
            success = true;
        } else {
            printf("optim failure: iter_max reached before convergence could be achieved.\n");
            printf("optim failure: returned best guess.\n");
            
            std::cout << "error: " << err << std::endl;
        }
    } else if (conv_failure_switch == 2) {
        if (err <= err_tol && iter <= iter_max) {
            out_vals = x_p;
            success = true;
        } else {
            printf("optim failure: iter_max reached before convergence could be achieved.\n");
            printf("optim failure: best guess:\n");

            arma::cout << x_p.t() << arma::endl;
            std::cout << "error: " << err << std::endl;
        }
    } else {
        printf("optim failure: unrecognized conv_failure_switch value.\n");
        success = false;
    }
    //
    if (settings_inp) {
        settings_inp->zero_values = opt_objfn(x_p,opt_data);
        settings_inp->opt_iter = iter;
        settings_inp->opt_err  = err;
    }
}

//

inline
void
error_reporting(arma::vec& out_vals, const arma::vec& x_p, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, arma::mat* hess_out, void* opt_data)> opt_objfn, void* opt_data,
                bool& success, const double err, const double err_tol, const int iter, const int iter_max, const int conv_failure_switch, algo_settings* settings_inp)
{
    std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> lam_objfn = [opt_objfn] (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data) 
    -> double 
    {
        return opt_objfn(vals_inp,grad_out,nullptr,opt_data);
    };

    //

    error_reporting(out_vals,x_p,lam_objfn,opt_data,success,err,err_tol,iter,iter_max,conv_failure_switch,settings_inp);
}