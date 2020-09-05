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
 * L-BFGS method for quasi-Newton-based non-linear optimization
 */

#ifndef _optim_lbfgs_HPP
#define _optim_lbfgs_HPP

/**
 * @brief The Limited Memory Variant of the BFGS Optimization Algorithm
 *
 * @param init_out_vals a column vector of initial values, which will be replaced by the solution upon successful completion of the optimization algorithm.
 * @param opt_objfn the function to be minimized, taking three arguments:
 *   - \c vals_inp a vector of inputs;
 *   - \c grad_out a vector to store the gradient; and
 *   - \c opt_data additional data passed to the user-provided function.
 * @param opt_data additional data passed to the user-provided function.
 *
 * @return a boolean value indicating successful completion of the optimization algorithm.
 */

bool
lbfgs(Vec_t& init_out_vals, 
      std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
      void* opt_data);

/**
 * @brief The Limited Memory Variant of the BFGS Optimization Algorithm
 *
 * @param init_out_vals a column vector of initial values, which will be replaced by the solution upon successful completion of the optimization algorithm.
 * @param opt_objfn the function to be minimized, taking three arguments:
 *   - \c vals_inp a vector of inputs;
 *   - \c grad_out a vector to store the gradient; and
 *   - \c opt_data additional data passed to the user-provided function.
 * @param opt_data additional data passed to the user-provided function.
 * @param settings parameters controlling the optimization routine.
 *
 * @return a boolean value indicating successful completion of the optimization algorithm.
 */

bool
lbfgs(Vec_t& init_out_vals, 
      std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
      void* opt_data, 
      algo_settings_t& settings);

//
// internal

namespace internal
{

bool
lbfgs_impl(Vec_t& init_out_vals, 
           std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
           void* opt_data, 
           algo_settings_t* settings_inp);

// algorithm 7.4 of Nocedal and Wright (2006)
inline
Vec_t
lbfgs_recur(Vec_t q, 
            const Mat_t& s_mat, 
            const Mat_t& y_mat, 
            const uint_t M)
{
    Vec_t alpha_vec(M);

    // forwards

    // double rho = 1.0;

    for (size_t i = 0; i < M; ++i) {
        double rho = 1.0 / OPTIM_MATOPS_DOT_PROD(y_mat.col(i),s_mat.col(i));
        alpha_vec(i) = rho * OPTIM_MATOPS_DOT_PROD(s_mat.col(i),q);

        q -= alpha_vec(i)*y_mat.col(i);
    }

    Vec_t r = q * ( OPTIM_MATOPS_DOT_PROD(s_mat.col(0),y_mat.col(0)) / OPTIM_MATOPS_DOT_PROD(y_mat.col(0),y_mat.col(0)) );

    // backwards

    // double beta = 1.0;

    for (int i = M - 1; i >= 0; i--) {
        double rho = 1.0 / OPTIM_MATOPS_DOT_PROD(y_mat.col(i),s_mat.col(i));
        double beta = rho * OPTIM_MATOPS_DOT_PROD(y_mat.col(i),r);

        r += (alpha_vec(i) - beta)*s_mat.col(i);
    }

    return r;
}

}

#endif
