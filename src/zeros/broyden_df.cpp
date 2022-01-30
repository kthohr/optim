/*################################################################################
  ##
  ##   Copyright (C) 2016-2022 Keith O'Hara
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
 * Li and Fukushima (2000) derivative-free variant of Broyden's method for solving systems of nonlinear equations
 */

#include "optim.hpp"

// [OPTIM_BEGIN]
optimlib_inline
bool
optim::internal::broyden_df_impl(
    ColVec_t& init_out_vals, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t* settings_inp
)
{
    // notation: 'p' stands for '+1'.
    
    bool success = false;

    const size_t n_vals = BMO_MATOPS_SIZE(init_out_vals);

    // Broyden settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int print_level = settings.print_level;
    const uint_t conv_failure_switch = settings.conv_failure_switch;
    
    const size_t iter_max = settings.iter_max;
    const fp_t rel_objfn_change_tol = settings.rel_objfn_change_tol;
    const fp_t rel_sol_change_tol = settings.rel_sol_change_tol;

    const fp_t rho     = settings.broyden_settings.par_rho; 
    const fp_t sigma_1 = settings.broyden_settings.par_sigma_1;
    const fp_t sigma_2 = settings.broyden_settings.par_sigma_2;

    // initialization

    ColVec_t x = init_out_vals;
    ColVec_t d = BMO_MATOPS_ZERO_COLVEC(n_vals);

    Mat_t B = BMO_MATOPS_EYE(n_vals); // initial approx. to Jacobian

    ColVec_t objfn_vec = opt_objfn(x,opt_data);

    fp_t rel_objfn_change = BMO_MATOPS_L2NORM(objfn_vec);

    OPTIM_BROYDEN_DF_TRACE(-1, rel_objfn_change, 0.0, x, d, objfn_vec, 0.0, d, d, B);

    if (rel_objfn_change <= rel_objfn_change_tol) {
        return true;
    }

    fp_t Fx = BMO_MATOPS_L2NORM(objfn_vec);

    //

    d = -objfn_vec; // step 1

    ColVec_t objfn_vec_p = opt_objfn(x + d,opt_data);

    fp_t Fx_p = BMO_MATOPS_L2NORM(objfn_vec_p);

    fp_t lambda;

    if (Fx_p <= rho*Fx - sigma_2*BMO_MATOPS_DOT_PROD(d,d)) {
        // step 2
        lambda = 1.0;
    } else {
        // step 3
        lambda = df_proc_1(x, d, sigma_1, 0, opt_objfn, opt_data);
    }

    ColVec_t x_p = x + lambda*d; // step 4

    ColVec_t s = x_p - x;
    ColVec_t y = objfn_vec_p - objfn_vec;

    rel_objfn_change = BMO_MATOPS_L2NORM( BMO_MATOPS_ARRAY_DIV_ARRAY( y, (BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ABS(objfn_vec), OPTIM_FPN_SMALL_NUMBER)) ) );
    fp_t rel_sol_change = BMO_MATOPS_L1NORM( BMO_MATOPS_ARRAY_DIV_ARRAY( s, (BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ABS(x), OPTIM_FPN_SMALL_NUMBER)) ) );

    // B += (y - B*s) * BMO_MATOPS_TRANSPOSE(s) / BMO_MATOPS_DOT_PROD(s,s); // step 5
    B += (s - B*y) * BMO_MATOPS_TRANSPOSE(y) / (BMO_MATOPS_DOT_PROD(y,y) + 1.0e-14);

    OPTIM_BROYDEN_DF_TRACE(0, rel_objfn_change, rel_sol_change, x_p, d, objfn_vec_p, lambda, s, y, B);

    if (rel_objfn_change <= rel_objfn_change_tol) {
        init_out_vals = x_p;
        return true;
    }

    //

    x = x_p;
    objfn_vec = objfn_vec_p;
    Fx = Fx_p;

    // begin loop

    size_t iter = 0;

    while (rel_objfn_change > rel_objfn_change_tol && rel_sol_change > rel_sol_change_tol && iter < iter_max) {
        ++iter;

        // d = arma::solve(B,-objfn_vec);
        d = - B*objfn_vec;

        objfn_vec_p = opt_objfn(x + d,opt_data);

        //

        Fx_p = BMO_MATOPS_L2NORM(objfn_vec_p);

        if (Fx_p <= rho*Fx - sigma_2*BMO_MATOPS_DOT_PROD(d,d)) {
            lambda = 1.0;
        } else {
            lambda = df_proc_1(x, d, sigma_1, iter, opt_objfn, opt_data);
        }

        //

        x_p = x + lambda*d;

        s = x_p - x;
        y = objfn_vec_p - objfn_vec;

        // B += (y - B*s) * s.t() / BMO_MATOPS_DOT_PROD(s,s);
        B += (s - B*y) * BMO_MATOPS_TRANSPOSE(y) / (BMO_MATOPS_DOT_PROD(y,y) + 1.0e-14); // update B

        //

        rel_objfn_change = BMO_MATOPS_L2NORM( BMO_MATOPS_ARRAY_DIV_ARRAY( y, (BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ABS(objfn_vec), OPTIM_FPN_SMALL_NUMBER)) ) );
        rel_sol_change = BMO_MATOPS_L1NORM( BMO_MATOPS_ARRAY_DIV_ARRAY( s, (BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ABS(x), OPTIM_FPN_SMALL_NUMBER)) ) );

        x = x_p;
        objfn_vec = objfn_vec_p;
        Fx = Fx_p;

        //

        OPTIM_BROYDEN_DF_TRACE(iter, rel_objfn_change, rel_sol_change, x_p, d, objfn_vec_p, lambda, s, y, B);
    }

    //

    error_reporting(init_out_vals,x_p,opt_objfn,opt_data,success,rel_objfn_change,rel_objfn_change_tol,iter,iter_max,conv_failure_switch,settings_inp);
    
    return success;
}

optimlib_inline
bool
optim::broyden_df(
    ColVec_t& init_out_vals, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data
)
{
    return internal::broyden_df_impl(init_out_vals,opt_objfn,opt_data,nullptr);
}

optimlib_inline
bool
optim::broyden_df(
    ColVec_t& init_out_vals, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t& settings
)
{
    return internal::broyden_df_impl(init_out_vals,opt_objfn,opt_data,&settings);
}

//
// derivative-free method with jacobian

optimlib_inline
bool
optim::internal::broyden_df_impl(
    ColVec_t& init_out_vals, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data,
    std::function<Mat_t (const ColVec_t& vals_inp, void* jacob_data)> jacob_objfn, 
    void* jacob_data, 
    algo_settings_t* settings_inp
)
{
    // notation: 'p' stands for '+1'.
    
    bool success = false;

    const size_t n_vals = BMO_MATOPS_SIZE(init_out_vals);

    // Broyden settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int print_level = settings.print_level;

    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const size_t iter_max = settings.iter_max;
    const fp_t rel_objfn_change_tol = settings.rel_objfn_change_tol;
    const fp_t rel_sol_change_tol = settings.rel_sol_change_tol;

    const fp_t rho     = settings.broyden_settings.par_rho; 
    const fp_t sigma_1 = settings.broyden_settings.par_sigma_1;
    const fp_t sigma_2 = settings.broyden_settings.par_sigma_2;

    // initialization

    ColVec_t x = init_out_vals;
    ColVec_t d = BMO_MATOPS_ZERO_COLVEC(n_vals);

    Mat_t B = BMO_MATOPS_INV( jacob_objfn(x, jacob_data) ); // inverse Jacobian

    ColVec_t objfn_vec = opt_objfn(x, opt_data);

    fp_t rel_objfn_change = BMO_MATOPS_L2NORM(objfn_vec);

    OPTIM_BROYDEN_DF_TRACE(-1, rel_objfn_change, 0.0, x, d, objfn_vec, 0.0, d, d, B);

    if (rel_objfn_change <= rel_objfn_change_tol) {
        return true;
    }

    fp_t Fx = BMO_MATOPS_L2NORM(objfn_vec);

    //

    d = - B * objfn_vec; // step 1

    ColVec_t objfn_vec_p = opt_objfn(x + d, opt_data);

    fp_t Fx_p = BMO_MATOPS_L2NORM(objfn_vec_p);

    fp_t lambda;

    if (Fx_p <= rho*Fx - sigma_2*BMO_MATOPS_DOT_PROD(d,d)) {
        // step 2
        lambda = 1.0;
    } else {
        // step 3
        lambda = df_proc_1(x, d, sigma_1, 0, opt_objfn, opt_data);
    }

    ColVec_t x_p = x + lambda*d; // step 4

    ColVec_t s = x_p - x;
    ColVec_t y = objfn_vec_p - objfn_vec;

    rel_objfn_change = BMO_MATOPS_L2NORM( BMO_MATOPS_ARRAY_DIV_ARRAY( y, (BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ABS(objfn_vec), OPTIM_FPN_SMALL_NUMBER)) ) );
    fp_t rel_sol_change = BMO_MATOPS_L1NORM( BMO_MATOPS_ARRAY_DIV_ARRAY( s, (BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ABS(x), OPTIM_FPN_SMALL_NUMBER)) ) );

    // B += (y - B*s) * s.t() / BMO_MATOPS_DOT_PROD(s,s); // step 5
    B += (s - B*y) * BMO_MATOPS_TRANSPOSE(y) / (BMO_MATOPS_DOT_PROD(y,y) + 1.0e-14); // update B

    OPTIM_BROYDEN_DF_TRACE(0, rel_objfn_change, rel_sol_change, x_p, d, objfn_vec_p, lambda, s, y, B);

    //

    x = x_p;
    objfn_vec = objfn_vec_p;
    Fx = Fx_p;

    // begin loop

    size_t iter = 0;

    while (rel_objfn_change > rel_objfn_change_tol && rel_sol_change > rel_sol_change_tol && iter < iter_max) {
        ++iter;

        // d = arma::solve(B,-objfn_vec);
        d = - B*objfn_vec;
        objfn_vec_p = opt_objfn(x + d,opt_data);

        //

        Fx_p = BMO_MATOPS_L2NORM(objfn_vec_p);

        if (Fx_p <= rho*Fx - sigma_2*BMO_MATOPS_DOT_PROD(d,d)) {
            lambda = 1.0;
        } else {
            lambda = df_proc_1(x,d,sigma_1,iter,opt_objfn,opt_data);
        }

        //

        x_p = x + lambda*d;

        s = x_p - x;
        y = objfn_vec_p - objfn_vec;

        if (iter % 5 == 0) {
            // B = jacob_objfn(x_p,jacob_data);
            B = BMO_MATOPS_INV( jacob_objfn(x_p,jacob_data) );
        } else {
            // B += (y - B*s) * s.t() / BMO_MATOPS_DOT_PROD(s,s);
            B += (s - B*y) * BMO_MATOPS_TRANSPOSE(y) / (BMO_MATOPS_DOT_PROD(y,y) + 1.0e-14); // update B
        }

        rel_objfn_change = BMO_MATOPS_L2NORM( BMO_MATOPS_ARRAY_DIV_ARRAY( y, (BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ABS(objfn_vec), OPTIM_FPN_SMALL_NUMBER)) ) );
        rel_sol_change = BMO_MATOPS_L1NORM( BMO_MATOPS_ARRAY_DIV_ARRAY( s, (BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ABS(x), OPTIM_FPN_SMALL_NUMBER)) ) );

        //

        x = x_p;
        objfn_vec = objfn_vec_p;
        Fx = Fx_p;

        //

        OPTIM_BROYDEN_DF_TRACE(iter, rel_objfn_change, rel_sol_change, x_p, d, objfn_vec_p, lambda, s, y, B);
    }

    //

    error_reporting(init_out_vals, x_p, opt_objfn, opt_data,
                    success, rel_objfn_change, rel_objfn_change_tol, 
                    iter, iter_max, conv_failure_switch, settings_inp);
    
    return success;
}

optimlib_inline
bool
optim::broyden_df(
    ColVec_t& init_out_vals, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data,
    std::function<Mat_t (const ColVec_t& vals_inp, void* jacob_data)> jacob_objfn, 
    void* jacob_data
)
{
    return internal::broyden_df_impl(init_out_vals,opt_objfn,opt_data,jacob_objfn,jacob_data,nullptr);
}

optimlib_inline
bool
optim::broyden_df(
    ColVec_t& init_out_vals, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data,
    std::function<Mat_t (const ColVec_t& vals_inp, void* jacob_data)> jacob_objfn, 
    void* jacob_data, 
    algo_settings_t& settings
)
{
    return internal::broyden_df_impl(init_out_vals,opt_objfn,opt_data,jacob_objfn,jacob_data,&settings);
}
