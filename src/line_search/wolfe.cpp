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
 *  Wolfe's method for line search
 *
 * – the starting point t_init of the line-search;
 * – the direction of search d;
 * – a merit-function t |-> q(t), defined for t >= 0, representing f(x + td).
 *
 * Keith O'Hara
 * 12/23/2016
 *
 * This version:
 * 01/01/2017
 */

#include "optim.hpp"

// a simple update
double optim::line_search_wolfe_simple(double t_init, const arma::vec& x, const arma::vec& d, double* c_1_inp, double* c_2_inp, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data)
{
    int n_vals = x.n_elem;

    // set Wolfe parameters
    double c_1 = (c_1_inp) ? *c_1_inp : 1E-04;
    double c_2 = (c_2_inp) ? *c_2_inp : 0.90;
    double a_t = 10; // extrapolation parameter

    double t_L = 0, t_R = 0; // initialization
    double t = t_init;
    //
    arma::vec grad_0(n_vals), grad_t(n_vals);

    double q_0 = opt_objfn(x,&grad_0,opt_data); // q(0)
    double q_deriv_0 = arma::dot(grad_0,d);    // q'(0)

    double q_t = opt_objfn(x + t*d,&grad_t,opt_data); // q(t)
    double q_deriv_t = arma::dot(grad_t,d);          // q'(t)
    //
    bool wolfe_cont = true;
    double check_term_1, check_term_2 = c_2*q_deriv_0;

    while (wolfe_cont) {
        q_t = opt_objfn(x + t*d,&grad_t,opt_data);
        q_deriv_t = arma::dot(grad_t,d);

        // test t
        check_term_1 = q_0 + c_1*t*q_deriv_0;

        if (q_t <= check_term_1 && q_deriv_t >= check_term_2) {
            wolfe_cont = false;
            break;
        } else if (q_t > check_term_1) {
            t_R = t;
        } else if (q_t <= check_term_1 && q_deriv_t < check_term_2) {
            t_L = t;
        }

        // compute new t
        if (t_R > 0) {
            t = (t_L + t_R) / 2; // simple update
        } else {
            t = a_t*t_L;
        }
    }
    //
    return t;
}

// cubic update
double optim::line_search_wolfe_cubic(double t_init, const arma::vec& x, const arma::vec& d, double* c_1_inp, double* c_2_inp, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data)
{
    int n_vals = x.n_elem;

    // set Wolfe parameters
    double c_1 = (c_1_inp) ? *c_1_inp : 1E-04;
    double c_2 = (c_2_inp) ? *c_2_inp : 0.90;
    double a_t = 10, theta = 0.25; // extrapolation and regularization parameters

    double t_L = 0, t_R = 0; // initialization
    double t = t_init, tm = 0, tp = t;
    //
    arma::vec grad_0(n_vals), grad_t(n_vals);

    double q_0 = opt_objfn(x,&grad_0,opt_data); // q(0)
    double q_deriv_0 = arma::dot(grad_0,d);    // q'(0)

    double q_t = opt_objfn(x + t*d,&grad_t,opt_data); // q(t)
    double q_deriv_t = arma::dot(grad_t,d);          // q'(t)

    double q_tm = q_0, q_deriv_tm = q_deriv_0;
    //
    bool wolfe_cont = true;
    int t_sign;
    double check_term_1, check_term_2 = c_2*q_deriv_0, p,D_1,D_2,r;

    while (wolfe_cont) {
        q_t = opt_objfn(x + t*d,&grad_t,opt_data);
        q_deriv_t = arma::dot(grad_t,d);

        // test t
        check_term_1 = q_0 + c_1*t*q_deriv_0;

        if (q_t <= check_term_1 && q_deriv_t >= check_term_2) {
            wolfe_cont = false;
            break;
        } else if (q_t > check_term_1) {
            t_R = t;
        } else if (q_t <= check_term_1 && q_deriv_t < check_term_2) {
            t_L = t;
        }

        // compute new t
        p = q_deriv_t + q_deriv_tm - 3*(q_t - q_tm)/(t - tm);
        D_1 = std::pow(p,2) - q_deriv_t*q_deriv_tm;

        if (D_1 > 0) {
            t_sign = (t - tm > 0) - (t - tm < 0);
            D_2 = std::sqrt(D_1)*t_sign;
        } else {
            D_1 = 0.0;
            D_2 = 0.0;
        }

        r = (D_2 + p - q_deriv_t) / (2*D_2 + q_deriv_tm - q_deriv_t);

        tp = t + r*(tm - t);
        // sanity check
        if (t_R > 0) {
            tp = std::min(tp,t_R - theta*(t_R-t_L));
            tp = std::max(tp,t_L + theta*(t_R-t_L));
        } else {
            tp = std::max(tp,a_t*t);
        }
        //
        tm = t;
        t = tp;

        q_tm = q_t;
        q_deriv_tm = q_deriv_t;
    }
    //
    return t;
}
