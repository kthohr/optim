/*################################################################################
  ##
  ##   Copyright (C) 2016-2020 Keith O'Hara
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

// [OPTIM_BEGIN]
optimlib_inline
bool
optim::broyden_int(Vec_t& init_out_vals, 
                   std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                   void* opt_data, 
                   algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.

    bool success = false;

    const size_t n_vals = OPTIM_MATOPS_SIZE(init_out_vals);

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

    Vec_t x = init_out_vals;

    Mat_t B = OPTIM_MATOPS_EYE(n_vals); // initial approx. to (inverse) Jacobian

    Vec_t f_val = opt_objfn(x,opt_data);

    double err = OPTIM_MATOPS_ACCU_ABS(f_val);
    if (err <= err_tol) {
        return true;
    }

    //

    Vec_t d = - B*f_val;

    Vec_t x_p = x + d;
    Vec_t f_val_p = opt_objfn(x_p,opt_data);

    err = OPTIM_MATOPS_ACCU_ABS(f_val_p);
    if (err <= err_tol) {
        init_out_vals = x_p;
        return true;
    }

    //

    Vec_t s = x_p - x;
    Vec_t y = f_val_p - f_val;

    B += (s - B*y) * OPTIM_MATOPS_TRANSPOSE(y) / OPTIM_MATOPS_DOT_PROD(y,y); // update B

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

        err = OPTIM_MATOPS_ACCU_ABS(f_val_p);

        if (err <= err_tol) {
            break;
        }

        //

        s = x_p - x;
        y = f_val_p - f_val;
        
        B += (s - B*y) * OPTIM_MATOPS_TRANSPOSE(y) / OPTIM_MATOPS_DOT_PROD(y,y); // update B

        //

        x = x_p;
        f_val = f_val_p;
    }

    //

    error_reporting(init_out_vals,x_p,opt_objfn,opt_data,success,err,err_tol,iter,iter_max,conv_failure_switch,settings_inp);

    return success;
}

optimlib_inline
bool
optim::broyden(Vec_t& init_out_vals, 
               std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
               void* opt_data)
{
    return broyden_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

optimlib_inline
bool
optim::broyden(Vec_t& init_out_vals, 
               std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
               void* opt_data, 
               algo_settings_t& settings)
{
    return broyden_int(init_out_vals,opt_objfn,opt_data,&settings);
}

//
// broyden with jacobian

optimlib_inline
bool
optim::broyden_int(Vec_t& init_out_vals, 
                   std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                   void* opt_data,
                   std::function<Mat_t (const Vec_t& vals_inp, void* jacob_data)> jacob_objfn, 
                   void* jacob_data, 
                   algo_settings_t* settings_inp)
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

    Vec_t x = init_out_vals;

    Mat_t B = OPTIM_MATOPS_INV( jacob_objfn(x,jacob_data) ); // initial approx. to (inverse) Jacobian

    Vec_t f_val = opt_objfn(x,opt_data);

    double err = OPTIM_MATOPS_ACCU_ABS(f_val);
    if (err <= err_tol) {
        return true;
    }

    //

    Vec_t d = - B*f_val;

    Vec_t x_p = x + d;
    Vec_t f_val_p = opt_objfn(x_p,opt_data);

    err = OPTIM_MATOPS_ACCU_ABS(f_val_p);
    if (err <= err_tol) {
        init_out_vals = x_p;
        return true;
    }

    //

    Vec_t s = x_p - x;
    Vec_t y = f_val_p - f_val;

    B += (s - B*y) * OPTIM_MATOPS_TRANSPOSE(y) / OPTIM_MATOPS_DOT_PROD(y,y); // update B

    f_val = f_val_p;

    //
    // begin loop

    uint_t iter = 0;

    while (err > err_tol && iter < iter_max) {
        iter++;

        //

        d = - B*f_val;

        x_p = x + d;
        f_val_p = opt_objfn(x_p,opt_data);

        err = OPTIM_MATOPS_ACCU_ABS(f_val_p);

        if (err <= err_tol) {
            break;
        }

        //

        s = x_p - x;
        y = f_val_p - f_val;
        
        if (iter % 5 == 0) {
            B = OPTIM_MATOPS_INV( jacob_objfn(x_p,jacob_data) );
        } else {
            B += (s - B*y) * OPTIM_MATOPS_TRANSPOSE(y) / OPTIM_MATOPS_DOT_PROD(y,y); // update B
        }

        //

        x = x_p;
        f_val = f_val_p;
    }

    //

    error_reporting(init_out_vals,x_p,opt_objfn,opt_data,success,err,err_tol,iter,iter_max,conv_failure_switch,settings_inp);

    return success;
}

optimlib_inline
bool
optim::broyden(Vec_t& init_out_vals, 
               std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
               void* opt_data,
               std::function<Mat_t (const Vec_t& vals_inp, void* jacob_data)> jacob_objfn, 
               void* jacob_data)
{
    return broyden_int(init_out_vals,opt_objfn,opt_data,jacob_objfn,jacob_data,nullptr);
}

optimlib_inline
bool
optim::broyden(Vec_t& init_out_vals, 
               std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
               void* opt_data,
               std::function<Mat_t (const Vec_t& vals_inp, void* jacob_data)> jacob_objfn, 
               void* jacob_data,
               algo_settings_t& settings)
{
    return broyden_int(init_out_vals,opt_objfn,opt_data,jacob_objfn,jacob_data,&settings);
}

//
// derivative-free method of Li and Fukushima (2000)

optimlib_inline
bool
optim::broyden_df_int(Vec_t& init_out_vals, 
                      std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                      void* opt_data, 
                      algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.
    
    bool success = false;

    const size_t n_vals = OPTIM_MATOPS_SIZE(init_out_vals);

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

    Vec_t x = init_out_vals;

    Mat_t B = OPTIM_MATOPS_EYE(n_vals); // initial approx. to Jacobian

    Vec_t f_val = opt_objfn(x,opt_data);
    double err = OPTIM_MATOPS_ACCU_ABS(f_val);

    if (err <= err_tol) {
        return true;
    }

    double Fx = OPTIM_MATOPS_L2NORM(f_val);

    //

    Vec_t d = -f_val; // step 1

    Vec_t f_val_p = opt_objfn(x + d,opt_data);
    err = OPTIM_MATOPS_ACCU_ABS(f_val_p);

    if (err <= err_tol)
    {
        init_out_vals = x + d;
        return true;
    }

    //

    double lambda;
    double Fx_p = OPTIM_MATOPS_L2NORM(f_val_p);

    if (Fx_p <= rho*Fx - sigma_2*OPTIM_MATOPS_DOT_PROD(d,d)) { // step 2
        lambda = 1.0;
    } else {
        lambda = df_proc_1(x,d,sigma_1,0,opt_objfn,opt_data); // step 3
    }

    //

    Vec_t x_p = x + lambda*d; // step 4

    Vec_t s = x_p - x;
    Vec_t y = f_val_p - f_val;

    // B += (y - B*s) * s.t() / OPTIM_MATOPS_DOT_PROD(s,s); // step 5
    B += (s - B*y) * OPTIM_MATOPS_TRANSPOSE(y) / OPTIM_MATOPS_DOT_PROD(y,y);

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

        err = OPTIM_MATOPS_ACCU_ABS(f_val);

        if (err <= err_tol) {
            break;
        }

        //

        Fx_p = OPTIM_MATOPS_L2NORM(f_val_p);

        if (Fx_p <= rho*Fx - sigma_2*OPTIM_MATOPS_DOT_PROD(d,d)) {
            lambda = 1.0;
        } else {
            lambda = df_proc_1(x,d,sigma_1,iter,opt_objfn,opt_data);
        }

        //

        x_p = x + lambda*d;

        Vec_t s = x_p - x;
        Vec_t y = f_val_p - f_val;

        // B += (y - B*s) * s.t() / OPTIM_MATOPS_DOT_PROD(s,s);
        B += (s - B*y) * OPTIM_MATOPS_TRANSPOSE(y) / OPTIM_MATOPS_DOT_PROD(y,y); // update B

        //

        x = x_p;
        f_val = f_val_p;
        Fx = Fx_p;
    }

    //

    error_reporting(init_out_vals,x_p,opt_objfn,opt_data,success,err,err_tol,iter,iter_max,conv_failure_switch,settings_inp);
    
    return success;
}

optimlib_inline
bool
optim::broyden_df(Vec_t& init_out_vals, 
                  std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                  void* opt_data)
{
    return broyden_df_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

optimlib_inline
bool
optim::broyden_df(Vec_t& init_out_vals, 
                  std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                  void* opt_data, 
                  algo_settings_t& settings)
{
    return broyden_df_int(init_out_vals,opt_objfn,opt_data,&settings);
}

//
// derivative-free method with jacobian

optimlib_inline
bool
optim::broyden_df_int(Vec_t& init_out_vals, 
                      std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                      void* opt_data,
                      std::function<Mat_t (const Vec_t& vals_inp, void* jacob_data)> jacob_objfn, 
                      void* jacob_data, 
                      algo_settings_t* settings_inp)
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

    Vec_t x = init_out_vals;

    Mat_t B = jacob_objfn(x,jacob_data); // Jacobian

    Vec_t f_val = opt_objfn(x,opt_data);
    double err = OPTIM_MATOPS_ACCU_ABS(f_val);

    if (err <= err_tol) {
        return true;
    }

    double Fx = OPTIM_MATOPS_L2NORM(f_val);

    //

    Vec_t d = OPTIM_MATOPS_SOLVE(B,-f_val); // step 1

    Vec_t f_val_p = opt_objfn(x + d,opt_data);
    err = OPTIM_MATOPS_ACCU_ABS(f_val_p);

    if (err <= err_tol)
    {
        init_out_vals = x + d;
        return true;
    }

    //

    double lambda;
    double Fx_p = OPTIM_MATOPS_L2NORM(f_val_p);

    if (Fx_p <= rho*Fx - sigma_2*OPTIM_MATOPS_DOT_PROD(d,d)) { // step 2
        lambda = 1.0;
    } else {
        lambda = df_proc_1(x,d,sigma_1,0,opt_objfn,opt_data); // step 3
    }

    //

    Vec_t x_p = x + lambda*d; // step 4

    Vec_t s = x_p - x;
    Vec_t y = f_val_p - f_val;

    B = OPTIM_MATOPS_INV(B); // switch to B^{-1}

    // B += (y - B*s) * s.t() / OPTIM_MATOPS_DOT_PROD(s,s); // step 5
    B += (s - B*y) * OPTIM_MATOPS_TRANSPOSE(y) / OPTIM_MATOPS_DOT_PROD(y,y); // update B

    //

    x = x_p;
    f_val = f_val_p;
    Fx = Fx_p;

    //
    // begin loop

    uint_t iter = 0;

    while (err > err_tol && iter < iter_max) {
        iter++;

        // d = arma::solve(B,-f_val);
        d = - B*f_val;
        f_val_p = opt_objfn(x + d,opt_data);

        err = OPTIM_MATOPS_ACCU_ABS(f_val);

        if (err <= err_tol) {
            break;
        }

        //

        Fx_p = OPTIM_MATOPS_L2NORM(f_val_p);

        if (Fx_p <= rho*Fx - sigma_2*OPTIM_MATOPS_DOT_PROD(d,d)) {
            lambda = 1.0;
        } else {
            lambda = df_proc_1(x,d,sigma_1,iter,opt_objfn,opt_data);
        }

        //

        x_p = x + lambda*d;

        Vec_t s = x_p - x;
        Vec_t y = f_val_p - f_val;

        if (iter % 5 == 0) {
            // B = jacob_objfn(x_p,jacob_data);
            B = OPTIM_MATOPS_INV( jacob_objfn(x_p,jacob_data) );
        } else {
            // B += (y - B*s) * s.t() / OPTIM_MATOPS_DOT_PROD(s,s);
            B += (s - B*y) * OPTIM_MATOPS_TRANSPOSE(y) / OPTIM_MATOPS_DOT_PROD(y,y); // update B
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

optimlib_inline
bool
optim::broyden_df(Vec_t& init_out_vals, 
                  std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                  void* opt_data,
                  std::function<Mat_t (const Vec_t& vals_inp, void* jacob_data)> jacob_objfn, 
                  void* jacob_data)
{
    return broyden_df_int(init_out_vals,opt_objfn,opt_data,jacob_objfn,jacob_data,nullptr);
}

optimlib_inline
bool
optim::broyden_df(Vec_t& init_out_vals, 
                  std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                  void* opt_data,
                  std::function<Mat_t (const Vec_t& vals_inp, void* jacob_data)> jacob_objfn, 
                  void* jacob_data, 
                  algo_settings_t& settings)
{
    return broyden_df_int(init_out_vals,opt_objfn,opt_data,jacob_objfn,jacob_data,&settings);
}

//
// internal functions

optimlib_inline
double
optim::df_eta(uint_t k)
{
    return 1.0 / (k*k);
}

optimlib_inline
double 
optim::df_proc_1(const Vec_t& x_vals, 
                 const Vec_t& direc, 
                 double sigma_1, 
                 uint_t k, 
                 std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                 void* opt_data)
{
    const double beta = 0.9;
    const double eta_k = df_eta(k);
    double lambda = 1.0;

    //
    // check: || F(x_k + lambda*d_k) || <= ||F(x_k)||*(1+eta_k) - sigma_1*||lambda*d_k||^2

    double Fx = OPTIM_MATOPS_L2NORM(opt_objfn(x_vals,opt_data));
    double Fx_p = OPTIM_MATOPS_L2NORM(opt_objfn(x_vals + lambda*direc,opt_data));
    double direc_norm2 = OPTIM_MATOPS_DOT_PROD(direc,direc);

    double term_2 = sigma_1*std::pow(lambda,2)*direc_norm2;
    double term_3 = eta_k*Fx;
    
    if (Fx_p <= Fx - term_2 + term_3) {
        return lambda;
    }

    //
    // begin loop

    uint_t iter = 0;
    
    while (1) {
        iter++;
        lambda *= beta; // lambda_i = beta^i;

        //

        Fx_p = OPTIM_MATOPS_L2NORM( opt_objfn(x_vals + lambda*direc,opt_data) );
        term_2 = sigma_1 * std::pow(lambda,2) * direc_norm2;

        if (Fx_p <= Fx - term_2 + term_3) {
            break;
        }
    }

    //

    return lambda;
}
