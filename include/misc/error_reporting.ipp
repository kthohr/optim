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
 * 07/19/2017
 */

inline
void
error_reporting(arma::vec& out_vals, const arma::vec& x_p, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data,
                bool& success, double* value_out, const double err, const double err_tol, const int iter, const int iter_max, const int conv_failure_switch)
{
    success = false;

    if (conv_failure_switch == 0) {
        out_vals = x_p;

        if (value_out) {
            *value_out = opt_objfn(x_p,nullptr,opt_data);
        }

        if (err <= err_tol && iter <= iter_max) {
            success = true;
        }
    } else if (conv_failure_switch == 1) {
        out_vals = x_p;

        if (value_out) {
            *value_out = opt_objfn(x_p,nullptr,opt_data);
        }

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

            if (value_out) {
                *value_out = opt_objfn(x_p,nullptr,opt_data);
            }
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
}

inline
void
error_reporting(arma::vec& out_vals, const arma::vec& x_p, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data,
                bool& success, double* value_out, const int conv_failure_switch)
{
    if (conv_failure_switch == 0 || conv_failure_switch == 1) {
        out_vals = x_p;

        if (value_out) {
            *value_out = opt_objfn(x_p,nullptr,opt_data);
        }
    } else if (conv_failure_switch == 2) {
        if (success) {
            out_vals = x_p;

            if (value_out) {
                *value_out = opt_objfn(x_p,nullptr,opt_data);
            }
        }
    } else {
        printf("optim failure: unrecognized conv_failure_switch value.\n");
        success = false;
    }
}

inline
void
error_reporting(arma::vec& out_vals, const arma::vec& x_p, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data,
                bool& success, arma::vec* value_out, const double err, const double err_tol, const int iter, const int iter_max, const int conv_failure_switch)
{
    success = false;

    if (conv_failure_switch == 0) {
        out_vals = x_p;

        if (value_out) {
            *value_out = opt_objfn(x_p,opt_data);
        }

        if (err <= err_tol && iter <= iter_max) {
            success = true;
        }
    } else if (conv_failure_switch == 1) {
        out_vals = x_p;

        if (value_out) {
            *value_out = opt_objfn(x_p,opt_data);
        }

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

            if (value_out) {
                *value_out = opt_objfn(x_p,opt_data);
            }
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
}
