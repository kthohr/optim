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
 * Broyden's method for solving systems of nonlinear equations
 */

#include "optim.hpp"

bool
optim::broyden_int(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.

    bool success = false;

    const size_t n_vals = init_out_vals.n_elem;

    //
    // Broyden settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const uint_t iter_max = settings.iter_max;
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
    if (err <= err_tol)
    {
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

    uint_t iter = 0;

    while (err > err_tol && iter < iter_max)
    {
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

    return success;
}

bool
optim::broyden(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data)
{
    return broyden_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

bool
optim::broyden(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t& settings)
{
    return broyden_int(init_out_vals,opt_objfn,opt_data,&settings);
}

//
// broyden with jacobian

bool
optim::broyden_int(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data,
                   std::function<arma::mat (const arma::vec& vals_inp, void* jacob_data)> jacob_objfn, void* jacob_data, algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.
    
    bool success = false;

    //
    // Broyden settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const uint_t iter_max = settings.iter_max;
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
    if (err <= err_tol)
    {
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

    uint_t iter = 0;

    while (err > err_tol && iter < iter_max)
    {
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
               algo_settings_t& settings)
{
    return broyden_int(init_out_vals,opt_objfn,opt_data,jacob_objfn,jacob_data,&settings);
}

//
// derivative-free method of Li and Fukushima (2000)

bool
optim::broyden_df_int(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.
    
    bool success = false;

    const size_t n_vals = init_out_vals.n_elem;

    //
    // Broyden settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const uint_t iter_max = settings.iter_max;
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

    if (err <= err_tol)
    {
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

    uint_t iter = 0;

    while (err > err_tol && iter < iter_max)
    {
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

    error_reporting(init_out_vals,x_p,opt_objfn,opt_data,success,err,err_tol,iter,iter_max,conv_failure_switch,settings_inp);
    
    return success;
}

bool
optim::broyden_df(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data)
{
    return broyden_df_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

bool
optim::broyden_df(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data, algo_settings_t& settings)
{
    return broyden_df_int(init_out_vals,opt_objfn,opt_data,&settings);
}

//
// derivative-free method with jacobian

bool
optim::broyden_df_int(arma::vec& init_out_vals, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data,
                      std::function<arma::mat (const arma::vec& vals_inp, void* jacob_data)> jacob_objfn, void* jacob_data, algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.
    
    bool success = false;

    //
    // Broyden settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const uint_t iter_max = settings.iter_max;
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

    if (err <= err_tol)
    {
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

    uint_t iter = 0;

    while (err > err_tol && iter < iter_max)
    {
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
               std::function<arma::mat (const arma::vec& vals_inp, void* jacob_data)> jacob_objfn, void* jacob_data, algo_settings_t& settings)
{
    return broyden_df_int(init_out_vals,opt_objfn,opt_data,jacob_objfn,jacob_data,&settings);
}

//
// internal functions

double
optim::df_eta(uint_t k)
{
    return 1.0 / (k*k);
}

double 
optim::df_proc_1(const arma::vec& x_vals, const arma::vec& direc, double sigma_1, uint_t k, std::function<arma::vec (const arma::vec& vals_inp, void* opt_data)> opt_objfn, void* opt_data)
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

    uint_t iter = 0;
    
    while (1) 
    {
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
