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
 * Broyden's method for solving systems of nonlinear equations
 *
 * Keith O'Hara
 * 01/03/2017
 *
 * This version:
 * 07/19/2017
 */

#include "optim.hpp"

bool
optim::broyden_int(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data, opt_settings* settings_inp)
{
    // notation: 'p' stands for '+1'.
    //
    bool success = false;

    const int n_vals = init_out_vals.n_elem;

    //
    // Broyden settings

    opt_settings settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int conv_failure_switch = settings.conv_failure_switch;
    const int iter_max = settings.iter_max;
    const double err_tol = settings.err_tol;

    //
    // initialization

    arma::vec x = init_out_vals;

    arma::mat B = arma::eye(n_vals,n_vals); // initial approx. to (inverse) Jacobian

    arma::vec f_val = opt_objfn(x,opt_data);

    double err = arma::accu(arma::abs(f_val));
    if (err <= err_tol) {
        return true;
    }
    //
    arma::vec d = - B*f_val;

    arma::vec x_p = x + d;
    arma::vec f_val_p = opt_objfn(x_p,opt_data);

    err = arma::accu(arma::abs(f_val_p));
    if (err <= err_tol) {
        init_out_vals = x_p;
        return true;
    }
    //
    arma::vec s = x_p - x;
    arma::vec y = f_val_p - f_val;

    B += (s - B*y) * y.t() / arma::dot(y,y); // update B

    f_val = f_val_p;
    //
    // begin loop
    int iter = 0;

    while (err > err_tol && iter < iter_max) {
        iter++;
        //
        d = - B*f_val;

        x_p = x + d;
        f_val_p = opt_objfn(x_p,opt_data);

        err = arma::accu(arma::abs(f_val_p));

        if (err <= err_tol) {
            break;
        }
        //
        s = x_p - x;
        y = f_val_p - f_val;
        
        B += (s - B*y) * y.t() / arma::dot(y,y); // update B
        //
        x = x_p;
        f_val = f_val_p;
    }
    //
    error_reporting(init_out_vals,x_p,opt_objfn,opt_data,success,err,err_tol,iter,iter_max,conv_failure_switch,settings_inp);
    //
    return success;
}

bool
optim::broyden(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data)
{
    return broyden_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

bool
optim::broyden(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data, opt_settings& settings)
{
    return broyden_int(init_out_vals,opt_objfn,opt_data,&settings);
}

//
// broyden with jacobian

bool
optim::broyden_int(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data,
                   std::function<arma::mat (const arma::vec& vals_inp, void* jacob_data)> jacob_objfn, void* jacob_data, opt_settings* settings_inp)
{
    // notation: 'p' stands for '+1'.
    //
    bool success = false;

    // const int n_vals = init_out_vals.n_elem;

    //
    // Broyden settings

    opt_settings settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int conv_failure_switch = settings.conv_failure_switch;
    const int iter_max = settings.iter_max;
    const double err_tol = settings.err_tol;

    //
    // initialization

    arma::vec x = init_out_vals;

    arma::mat B = arma::inv(jacob_objfn(x,jacob_data)); // initial approx. to (inverse) Jacobian

    arma::vec f_val = opt_objfn(x,opt_data);

    double err = arma::accu(arma::abs(f_val));
    if (err <= err_tol) {
        return true;
    }
    //
    arma::vec d = - B*f_val;

    arma::vec x_p = x + d;
    arma::vec f_val_p = opt_objfn(x_p,opt_data);

    err = arma::accu(arma::abs(f_val_p));
    if (err <= err_tol) {
        init_out_vals = x_p;
        return true;
    }
    //
    arma::vec s = x_p - x;
    arma::vec y = f_val_p - f_val;

    B += (s - B*y) * y.t() / arma::dot(y,y); // update B

    f_val = f_val_p;
    //
    // begin loop
    int iter = 0;

    while (err > err_tol && iter < iter_max) {
        iter++;
        //
        d = - B*f_val;

        x_p = x + d;
        f_val_p = opt_objfn(x_p,opt_data);

        err = arma::accu(arma::abs(f_val_p));

        if (err <= err_tol) {
            break;
        }
        //
        s = x_p - x;
        y = f_val_p - f_val;
        
        if (iter % 5 == 0) {
            B = arma::inv(jacob_objfn(x_p,jacob_data));
        } else {
            B += (s - B*y) * y.t() / arma::dot(y,y); // update B
        }
        //
        x = x_p;
        f_val = f_val_p;
    }
    //
    error_reporting(init_out_vals,x_p,opt_objfn,opt_data,success,err,err_tol,iter,iter_max,conv_failure_switch,settings_inp);
    //
    return success;
}

bool
optim::broyden(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data,
               std::function<arma::mat (const arma::vec& vals_inp, void* jacob_data)> jacob_objfn, void* jacob_data)
{
    return broyden_int(init_out_vals,opt_objfn,opt_data,jacob_objfn,jacob_data,nullptr);
}

bool
optim::broyden(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data,
               std::function<arma::mat (const arma::vec& vals_inp, void* jacob_data)> jacob_objfn, void* jacob_data,
               opt_settings& settings)
{
    return broyden_int(init_out_vals,opt_objfn,opt_data,jacob_objfn,jacob_data,&settings);
}

//
// derivative-free method of Li and Fukushima (2000)

bool
optim::broyden_df_int(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data, opt_settings* settings_inp)
{
    // notation: 'p' stands for '+1'.
    //
    bool success = false;

    const int n_vals = init_out_vals.n_elem;

    //
    // Broyden settings

    opt_settings settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int conv_failure_switch = settings.conv_failure_switch;
    const int iter_max = settings.iter_max;
    const double err_tol = settings.err_tol;

    const double rho = 0.9, sigma_1 = 0.001, sigma_2 = 0.001; // tuning parameters

    //
    // initialization

    arma::vec x = init_out_vals;

    arma::mat B = arma::eye(n_vals,n_vals); // initial approx. to Jacobian

    arma::vec f_val = opt_objfn(x,opt_data);
    double err = arma::accu(arma::abs(f_val));

    if (err <= err_tol) {
        return true;
    }

    double Fx = arma::norm(f_val,2);
    //
    arma::vec d = -f_val; // step 1

    arma::vec f_val_p = opt_objfn(x + d,opt_data);
    err = arma::accu(arma::abs(f_val_p));

    if (err <= err_tol) {
        init_out_vals = x + d;
        return true;
    }
    //
    double lambda;
    double Fx_p = arma::norm(f_val_p,2);

    if (Fx_p <= rho*Fx - sigma_2*arma::dot(d,d)) { // step 2
        lambda = 1.0;
    } else {
        lambda = df_proc_1(x,d,sigma_1,0,opt_objfn,opt_data); // step 3
    }
    //
    arma::vec x_p = x + lambda*d; // step 4

    arma::vec s = x_p - x;
    arma::vec y = f_val_p - f_val;

    // B += (y - B*s) * s.t() / arma::dot(s,s); // step 5
    B += (s - B*y) * y.t() / arma::dot(y,y);
    //
    x = x_p;
    f_val = f_val_p;
    Fx = Fx_p;

    //
    // begin loop

    int iter = 0;

    while (err > err_tol && iter < iter_max) {
        iter++;

        // d = arma::solve(B,-f_val);
        d = - B*f_val;
        f_val_p = opt_objfn(x + d,opt_data);

        err = arma::accu(arma::abs(f_val));

        if (err <= err_tol) {
            break;
        }
        //
        Fx_p = arma::norm(f_val_p,2);

        if (Fx_p <= rho*Fx - sigma_2*arma::dot(d,d)) {
            lambda = 1.0;
        } else {
            lambda = df_proc_1(x,d,sigma_1,iter,opt_objfn,opt_data);
        }
        //
        x_p = x + lambda*d;

        arma::vec s = x_p - x;
        arma::vec y = f_val_p - f_val;

        // B += (y - B*s) * s.t() / arma::dot(s,s);
        B += (s - B*y) * y.t() / arma::dot(y,y); // update B
        //
        x = x_p;
        f_val = f_val_p;
        Fx = Fx_p;
    }

    //
    // end

    error_reporting(init_out_vals,x_p,opt_objfn,opt_data,success,err,err_tol,iter,iter_max,conv_failure_switch,settings_inp);
    
    return success;
}

bool
optim::broyden_df(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data)
{
    return broyden_df_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

bool
optim::broyden_df(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data, opt_settings& settings)
{
    return broyden_df_int(init_out_vals,opt_objfn,opt_data,&settings);
}

//
// derivative-free method with jacobian

bool
optim::broyden_df_int(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data,
                      std::function<arma::mat (const arma::vec& vals_inp, void* jacob_data)> jacob_objfn, void* jacob_data, opt_settings* settings_inp)
{
    // notation: 'p' stands for '+1'.
    //
    bool success = false;

    // const int n_vals = init_out_vals.n_elem;

    //
    // Broyden settings

    opt_settings settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int conv_failure_switch = settings.conv_failure_switch;
    const int iter_max = settings.iter_max;
    const double err_tol = settings.err_tol;

    const double rho = 0.9, sigma_1 = 0.001, sigma_2 = 0.001; // tuning parameters

    //
    // initialization

    arma::vec x = init_out_vals;

    arma::mat B = jacob_objfn(x,jacob_data); // Jacobian

    arma::vec f_val = opt_objfn(x,opt_data);
    double err = arma::accu(arma::abs(f_val));

    if (err <= err_tol) {
        return true;
    }

    double Fx = arma::norm(f_val,2);
    //
    arma::vec d = arma::solve(B,-f_val); // step 1

    arma::vec f_val_p = opt_objfn(x + d,opt_data);
    err = arma::accu(arma::abs(f_val_p));

    if (err <= err_tol) {
        init_out_vals = x + d;
        return true;
    }
    //
    double lambda;
    double Fx_p = arma::norm(f_val_p,2);

    if (Fx_p <= rho*Fx - sigma_2*arma::dot(d,d)) { // step 2
        lambda = 1.0;
    } else {
        lambda = df_proc_1(x,d,sigma_1,0,opt_objfn,opt_data); // step 3
    }
    //
    arma::vec x_p = x + lambda*d; // step 4

    arma::vec s = x_p - x;
    arma::vec y = f_val_p - f_val;

    B = arma::inv(B); // switch to B^{-1}

    // B += (y - B*s) * s.t() / arma::dot(s,s); // step 5
    B += (s - B*y) * y.t() / arma::dot(y,y); // update B
    //
    x = x_p;
    f_val = f_val_p;
    Fx = Fx_p;

    //
    // begin loop

    int iter = 0;

    while (err > err_tol && iter < iter_max) {
        iter++;

        // d = arma::solve(B,-f_val);
        d = - B*f_val;
        f_val_p = opt_objfn(x + d,opt_data);

        err = arma::accu(arma::abs(f_val));

        if (err <= err_tol) {
            break;
        }
        //
        Fx_p = arma::norm(f_val_p,2);

        if (Fx_p <= rho*Fx - sigma_2*arma::dot(d,d)) {
            lambda = 1.0;
        } else {
            lambda = df_proc_1(x,d,sigma_1,iter,opt_objfn,opt_data);
        }
        //
        x_p = x + lambda*d;

        arma::vec s = x_p - x;
        arma::vec y = f_val_p - f_val;

        if (iter % 5 == 0) {
            // B = jacob_objfn(x_p,jacob_data);
            B = arma::inv(jacob_objfn(x_p,jacob_data));
        } else {
            // B += (y - B*s) * s.t() / arma::dot(s,s);
            B += (s - B*y) * y.t() / arma::dot(y,y); // update B
        }
        //
        x = x_p;
        f_val = f_val_p;
        Fx = Fx_p;
    }

    //
    // end

    error_reporting(init_out_vals,x_p,opt_objfn,opt_data,success,err,err_tol,iter,iter_max,conv_failure_switch,settings_inp);
    
    return success;
}

bool
optim::broyden_df(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data,
               std::function<arma::mat (const arma::vec& vals_inp, void* jacob_data)> jacob_objfn, void* jacob_data)
{
    return broyden_df_int(init_out_vals,opt_objfn,opt_data,jacob_objfn,jacob_data,nullptr);
}

bool
optim::broyden_df(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data,
               std::function<arma::mat (const arma::vec& vals_inp, void* jacob_data)> jacob_objfn, void* jacob_data, opt_settings& settings)
{
    return broyden_df_int(init_out_vals,opt_objfn,opt_data,jacob_objfn,jacob_data,&settings);
}

//
// internal functions

double
optim::df_eta(int k)
{
    return 1.0 / (k*k);
}

double 
optim::df_proc_1(const arma::vec& x_vals, const arma::vec& direc, double sigma_1, int k, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data)
{
    const double beta = 0.9;
    const double eta_k = df_eta(k);
    double lambda = 1.0;

    //
    // check: || F(x_k + lambda*d_k) || <= ||F(x_k)||*(1+eta_k) - sigma_1*||lambda*d_k||^2

    double Fx = arma::norm(opt_objfn(x_vals,opt_data),2);
    double Fx_p = arma::norm(opt_objfn(x_vals + lambda*direc,opt_data),2);
    double direc_norm2 = arma::dot(direc,direc);

    double term_2 = sigma_1*std::pow(lambda,2)*direc_norm2;
    double term_3 = eta_k*Fx;
    
    if (Fx_p <= Fx - term_2 + term_3) {
        return lambda;
    }

    //
    // begin loop

    int iter = 0;
    
    while (1) {
        iter++;
        lambda *= beta; // lambda_i = beta^i;
        //
        Fx_p = arma::norm(opt_objfn(x_vals + lambda*direc,opt_data),2);
        term_2 = sigma_1*std::pow(lambda,2)*direc_norm2;

        if (Fx_p <= Fx - term_2 + term_3) {
            break;
        }
    }
    //
    return lambda;
}

/*
bool optim::broyden_mt(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data)
{
    // notation: 'p' stands for '+1'.
    //
    bool success = false;
    int iter_max = 1000;
    double err_tol = 1e-08;
    double wolfe_cons_1 = 1E-03;
    double wolfe_cons_2 = 0.50;
    //
    int n_vals = init_out_vals.n_elem;

    arma::vec x = init_out_vals;

    arma::mat B = arma::eye(n_vals,n_vals); // initial approx. to (inverse) Jacobian
    //
    // std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> ls_objfn = [opt_objfn, &B] (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data) -> double {
    //     int n_vals = vals_inp.n_elem;
    //     double eps_diff = 1e-04;

    //     arma::vec Fx = opt_objfn(vals_inp,opt_data);
    //     double ret = arma::dot(Fx,Fx)/2.0;
    //     std::cout << "lambda ret: " << ret << std::endl;

    //     arma::vec Fx_p, Fx_m;
    //     for (int jj=0; jj < n_vals; jj++) {
    //         arma::vec Fx_p = opt_objfn(vals_inp + eps_diff*unit_vec(jj,n_vals),opt_data);
    //         arma::vec Fx_m = opt_objfn(vals_inp - eps_diff*unit_vec(jj,n_vals),opt_data);

    //         grad(jj) = (arma::dot(Fx_p,Fx_p)/2.0 - arma::dot(Fx_m,Fx_m)/2.0) / (2*eps_diff);
    //     }
    //     arma::cout << "lambda grad: " << grad.t() << arma::endl;
    //     //
    //     return ret;
    // };
    std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> ls_objfn = [opt_objfn, &B] (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data) -> double {
        int n_vals = vals_inp.n_elem;
        double eps_diff = 1e-06;

        arma::vec Fx = opt_objfn(vals_inp,opt_data);
        double ret = arma::dot(Fx,Fx)/2.0;
        std::cout << "lambda ret: " << ret << std::endl;

        if (grad_out) {
            arma::vec Fx_p, Fx_m;
            for (int jj=0; jj < n_vals; jj++) {
                arma::vec Fx_p = opt_objfn(vals_inp + eps_diff*unit_vec(jj,n_vals),opt_data);
                //arma::vec Fx_m = opt_objfn(vals_inp - eps_diff*unit_vec(jj,n_vals),opt_data);
                
                //grad(jj) = (arma::dot(Fx_p,Fx_p)/2.0 - arma::dot(Fx_m,Fx_m)/2.0) / (2*eps_diff);
                (*grad_out)(jj) = (arma::dot(Fx_p,Fx_p)/2.0 - arma::dot(Fx,Fx)/2.0) / (eps_diff);
            }
            arma::cout << "lambda grad: " << (*grad_out).t() << arma::endl;
        }
        //
        return ret;
    };
    //
    // initialization
    double t_init = 1;

    arma::vec f_val = opt_objfn(x,opt_data);
    double err = arma::accu(arma::abs(f_val));

    if (err <= err_tol) {
        return true;
    }
    //
    arma::vec d = - B*f_val;

    arma::vec x_p = x, grad_mt(n_vals);
    double t = line_search_mt(t_init, x_p, grad_mt, d, &wolfe_cons_1, &wolfe_cons_2, ls_objfn, opt_data);
    std::cout << "t: " << t << std::endl;
    arma::cout << x_p << arma::endl;

    arma::vec f_val_p = opt_objfn(x_p,opt_data);
    err = arma::accu(arma::abs(f_val_p));

    if (err <= err_tol) {
        init_out_vals = x_p;
        return true;
    }
    //
    arma::vec s = x_p - x;
    arma::vec y = f_val_p - f_val;

    // update B
    B += (s - B*y) * y.t() / arma::dot(y,y);

    f_val = f_val_p;
    //
    // begin loop
    int iter = 0;

    while (err > err_tol && iter < iter_max) {
        iter++;

        d = - B*f_val;
        arma::cout << "direction: \n" << d.t() << arma::endl;
        t = line_search_mt(t_init, x_p, grad_mt, d, &wolfe_cons_1, &wolfe_cons_2, ls_objfn, opt_data);

        f_val_p = opt_objfn(x_p,opt_data);
        err = arma::accu(arma::abs(f_val_p));

        if (err <= err_tol) {
            break;
        }

        s = x_p - x;
        y = f_val_p - f_val;
        // update B
        B += (s - B*y) * y.t() / arma::dot(y,y);
        //
        x = x_p;
        f_val = f_val_p;

        std::cout << "f_val_p:" << f_val_p << std::endl;
    }
    //
    if (err <= err_tol && iter <= iter_max) {
        init_out_vals = x_p;
        success = true;
    }
    //
    return success;
}
*/
