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
optim::internal::broyden_impl(
    Vec_t& init_out_vals, 
    std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.

    bool success = false;

    const size_t n_vals = OPTIM_MATOPS_SIZE(init_out_vals);

    // Broyden settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int print_level = settings.print_level;

    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const size_t iter_max = settings.iter_max;
    const double rel_objfn_change_tol = settings.rel_objfn_change_tol;
    const double rel_sol_change_tol = settings.rel_sol_change_tol;

    // initialization

    Vec_t x = init_out_vals;
    Vec_t d = OPTIM_MATOPS_ZERO_VEC(n_vals);

    Mat_t B = OPTIM_MATOPS_EYE(n_vals); // initial approx. to (inverse) Jacobian

    Vec_t objfn_vec = opt_objfn(x, opt_data);

    double rel_objfn_change = OPTIM_MATOPS_L2NORM(objfn_vec);

    OPTIM_BROYDEN_TRACE(-1, rel_objfn_change, 0.0, x, d, objfn_vec, d, d, B);

    if (rel_objfn_change <= rel_objfn_change_tol) {
        return true;
    }

    //

    d = - B*objfn_vec;
    Vec_t x_p = x + d;

    Vec_t objfn_vec_p = opt_objfn(x_p, opt_data);

    Vec_t s = x_p - x;
    Vec_t y = objfn_vec_p - objfn_vec;

    rel_objfn_change = OPTIM_MATOPS_L2NORM( OPTIM_MATOPS_ARRAY_DIV_ARRAY( y, (OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_ABS(objfn_vec), 1.0e-08)) ) );
    double rel_sol_change = OPTIM_MATOPS_L1NORM( OPTIM_MATOPS_ARRAY_DIV_ARRAY( s, (OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_ABS(x), 1.0e-08)) ) );

    B += (s - B*y) * OPTIM_MATOPS_TRANSPOSE(y) / (OPTIM_MATOPS_DOT_PROD(y,y) + 1.0e-14); // update B

    OPTIM_BROYDEN_TRACE(0, rel_objfn_change, rel_sol_change, x_p, d, objfn_vec_p, y, s, B);

    if (rel_objfn_change <= rel_objfn_change_tol) {
        init_out_vals = x_p;
        return true;
    }

    objfn_vec = objfn_vec_p;

    // begin loop

    size_t iter = 0;

    while (rel_objfn_change > rel_objfn_change_tol && rel_sol_change > rel_sol_change_tol && iter < iter_max) {
        ++iter;

        //

        d = - B*objfn_vec;
        x_p = x + d;
        
        objfn_vec_p = opt_objfn(x_p, opt_data);

        //

        s = x_p - x;
        y = objfn_vec_p - objfn_vec;
        
        B += (s - B*y) * OPTIM_MATOPS_TRANSPOSE(y) / (OPTIM_MATOPS_DOT_PROD(y,y) + 1.0e-14); // update B

        //

        rel_objfn_change = OPTIM_MATOPS_L2NORM( OPTIM_MATOPS_ARRAY_DIV_ARRAY( y, (OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_ABS(objfn_vec), 1.0e-08)) ) );
        rel_sol_change = OPTIM_MATOPS_L1NORM( OPTIM_MATOPS_ARRAY_DIV_ARRAY( s, (OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_ABS(x), 1.0e-08)) ) );

        x = x_p;
        objfn_vec = objfn_vec_p;

        //

        OPTIM_BROYDEN_TRACE(iter, rel_objfn_change, rel_sol_change, x, d, objfn_vec_p, s, y, B);
    }

    //

    error_reporting(init_out_vals, x_p, opt_objfn, opt_data,
                    success, rel_objfn_change, rel_objfn_change_tol, 
                    iter, iter_max, conv_failure_switch, settings_inp);

    return success;
}

optimlib_inline
bool
optim::broyden(
    Vec_t& init_out_vals, 
    std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data)
{
    return internal::broyden_impl(init_out_vals,opt_objfn,opt_data,nullptr);
}

optimlib_inline
bool
optim::broyden(Vec_t& init_out_vals, 
               std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
               void* opt_data, 
               algo_settings_t& settings)
{
    return internal::broyden_impl(init_out_vals,opt_objfn,opt_data,&settings);
}

//
// broyden with jacobian

optimlib_inline
bool
optim::internal::broyden_impl(
    Vec_t& init_out_vals, 
    std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data,
    std::function<Mat_t (const Vec_t& vals_inp, void* jacob_data)> jacob_objfn, 
    void* jacob_data, 
    algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.
    
    bool success = false;

    const size_t n_vals = OPTIM_MATOPS_SIZE(init_out_vals);

    // Broyden settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int print_level = settings.print_level;

    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const size_t iter_max = settings.iter_max;
    const double rel_objfn_change_tol = settings.rel_objfn_change_tol;
    const double rel_sol_change_tol = settings.rel_sol_change_tol;

    // initialization

    Vec_t x = init_out_vals;
    Vec_t d = OPTIM_MATOPS_ZERO_VEC(n_vals);

    Mat_t B = OPTIM_MATOPS_INV( jacob_objfn(x, jacob_data) ); // initial approx. to (inverse) Jacobian

    Vec_t objfn_vec = opt_objfn(x, opt_data);

    double rel_objfn_change = OPTIM_MATOPS_L2NORM(objfn_vec);

    OPTIM_BROYDEN_TRACE(-1, rel_objfn_change, 0.0, x, d, objfn_vec, d, d, B);
    
    if (rel_objfn_change <= rel_objfn_change_tol) {
        return true;
    }

    //

    d = - B*objfn_vec;
    Vec_t x_p = x + d;

    Vec_t objfn_vec_p = opt_objfn(x_p, opt_data);

    Vec_t s = x_p - x;
    Vec_t y = objfn_vec_p - objfn_vec;

    rel_objfn_change = OPTIM_MATOPS_L2NORM( OPTIM_MATOPS_ARRAY_DIV_ARRAY( y, (OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_ABS(objfn_vec), 1.0e-08)) ) );
    double rel_sol_change = OPTIM_MATOPS_L1NORM( OPTIM_MATOPS_ARRAY_DIV_ARRAY( s, (OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_ABS(x), 1.0e-08)) ) );

    B += (s - B*y) * OPTIM_MATOPS_TRANSPOSE(y) / (OPTIM_MATOPS_DOT_PROD(y,y) + 1.0e-14); // update B

    OPTIM_BROYDEN_TRACE(0, rel_objfn_change, rel_sol_change, x_p, d, objfn_vec, y, s, B);

    if (rel_objfn_change <= rel_objfn_change_tol) {
        init_out_vals = x_p;
        return true;
    }

    objfn_vec = objfn_vec_p;

    // begin loop

    size_t iter = 0;

    while (rel_objfn_change > rel_objfn_change_tol && rel_sol_change > rel_sol_change_tol && iter < iter_max) {
        ++iter;

        //

        d = - B*objfn_vec;

        x_p = x + d;
        objfn_vec_p = opt_objfn(x_p,opt_data);

        //

        s = x_p - x;
        y = objfn_vec_p - objfn_vec;
        
        if (iter % 5 == 0) {
            B = OPTIM_MATOPS_INV( jacob_objfn(x_p,jacob_data) );
        } else {
            B += (s - B*y) * OPTIM_MATOPS_TRANSPOSE(y) / (OPTIM_MATOPS_DOT_PROD(y,y) + 1.0e-14); // update B
        }

        //

        rel_objfn_change = OPTIM_MATOPS_L2NORM( OPTIM_MATOPS_ARRAY_DIV_ARRAY( y, (OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_ABS(objfn_vec), 1.0e-08)) ) );
        rel_sol_change = OPTIM_MATOPS_L1NORM( OPTIM_MATOPS_ARRAY_DIV_ARRAY( s, (OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_ABS(x), 1.0e-08)) ) );

        //

        x = x_p;
        objfn_vec = objfn_vec_p;

        //

        OPTIM_BROYDEN_TRACE(iter, rel_objfn_change, rel_sol_change, x, d, objfn_vec, s, y, B);
    }

    //

    error_reporting(init_out_vals, x_p, opt_objfn, opt_data,
                    success, rel_objfn_change, rel_objfn_change_tol, 
                    iter, iter_max, conv_failure_switch, settings_inp);

    return success;
}

optimlib_inline
bool
optim::broyden(
    Vec_t& init_out_vals, 
    std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data,
    std::function<Mat_t (const Vec_t& vals_inp, void* jacob_data)> jacob_objfn, 
    void* jacob_data)
{
    return internal::broyden_impl(init_out_vals, opt_objfn, opt_data, jacob_objfn, jacob_data, nullptr);
}

optimlib_inline
bool
optim::broyden(
    Vec_t& init_out_vals, 
    std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data,
    std::function<Mat_t (const Vec_t& vals_inp, void* jacob_data)> jacob_objfn, 
    void* jacob_data,
    algo_settings_t& settings)
{
    return internal::broyden_impl(init_out_vals, opt_objfn, opt_data, jacob_objfn, jacob_data, &settings);
}
