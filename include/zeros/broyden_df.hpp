/*################################################################################
  ##
  ##   Copyright (C) 2016-2023 Keith O'Hara
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

#ifndef _optim_broyden_df_HPP
#define _optim_broyden_df_HPP

/**
 * @brief Derivative-free variant of Broyden's method due to Li and Fukushima (2000)
 *
 * @param init_out_vals a column vector of initial values, which will be replaced by the solution upon successful completion of the optimization algorithm.
 * @param opt_objfn the function to be minimized, taking three arguments:
 *   - \c vals_inp a vector of inputs; and
 *   - \c opt_data additional data passed to the user-provided function.
 * @param opt_data additional data passed to the user-provided function.
 *
 * @return a boolean value indicating successful completion of the optimization algorithm.
 */

bool
broyden_df(
    ColVec_t& init_out_vals, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data
);

/**
 * @brief Derivative-free variant of Broyden's method due to Li and Fukushima (2000)
 *
 * @param init_out_vals a column vector of initial values, which will be replaced by the solution upon successful completion of the optimization algorithm.
 * @param opt_objfn the function to be minimized, taking three arguments:
 *   - \c vals_inp a vector of inputs; and
 *   - \c opt_data additional data passed to the user-provided function.
 * @param opt_data additional data passed to the user-provided function.
 * @param settings parameters controlling the optimization routine.
 *
 * @return a boolean value indicating successful completion of the optimization algorithm.
 */

bool
broyden_df(
    ColVec_t& init_out_vals, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t& settings
);

// derivative-free method with jacobian

/**
 * @brief Derivative-free variant of Broyden's method due to Li and Fukushima (2000)
 *
 * @param init_out_vals a column vector of initial values, which will be replaced by the solution upon successful completion of the optimization algorithm.
 * @param opt_objfn the function to be minimized, taking three arguments:
 *   - \c vals_inp a vector of inputs; and
 *   - \c opt_data additional data passed to the user-provided function.
 * @param opt_data additional data passed to the user-provided function.
 * @param jacob_objfn a function to calculate the Jacobian matrix, taking two arguments:
 *   - \c vals_inp a vector of inputs; and
 *   - \c jacob_data additional data passed to the Jacobian function.
 * @param jacob_data additional data passed to the Jacobian function.
 *
 * @return a boolean value indicating successful completion of the optimization algorithm.
 */

bool
broyden_df(
    ColVec_t& init_out_vals, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data,
    std::function<Mat_t (const ColVec_t& vals_inp, void* jacob_data)> jacob_objfn, 
    void* jacob_data
);

/**
 * @brief Derivative-free variant of Broyden's method due to Li and Fukushima (2000)
 *
 * @param init_out_vals a column vector of initial values, which will be replaced by the solution upon successful completion of the optimization algorithm.
 * @param opt_objfn the function to be minimized, taking three arguments:
 *   - \c vals_inp a vector of inputs; and
 *   - \c opt_data additional data passed to the user-provided function.
 * @param opt_data additional data passed to the user-provided function.
 * @param jacob_objfn a function to calculate the Jacobian matrix, taking two arguments:
 *   - \c vals_inp a vector of inputs; and
 *   - \c jacob_data additional data passed to the Jacobian function.
 * @param jacob_data additional data passed to the Jacobian function.
 * @param settings parameters controlling the optimization routine.
 *
 * @return a boolean value indicating successful completion of the optimization algorithm.
 */

bool
broyden_df(
    ColVec_t& init_out_vals, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data,
    std::function<Mat_t (const ColVec_t& vals_inp, void* jacob_data)> jacob_objfn, 
    void* jacob_data, 
    algo_settings_t& settings
);

//
// internal functions

namespace internal
{

bool 
broyden_df_impl(
    ColVec_t& init_out_vals, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t* settings_inp
);

bool
broyden_df_impl(
    ColVec_t& init_out_vals, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data,
    std::function<Mat_t (const ColVec_t& vals_inp, void* jacob_data)> jacob_objfn, 
    void* jacob_data, 
    algo_settings_t* settings_inp
);

//

inline
fp_t
df_eta(uint_t k)
{
    return 1.0 / (k*k);
}

inline
fp_t 
df_proc_1(
    const ColVec_t& x_vals, 
    const ColVec_t& direc, 
    fp_t sigma_1, 
    uint_t k, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data
)
{
    const fp_t beta = 0.9;
    const fp_t eta_k = df_eta(k);
    fp_t lambda = 1.0;

    // check: || F(x_k + lambda*d_k) || <= ||F(x_k)||*(1+eta_k) - sigma_1*||lambda*d_k||^2

    fp_t Fx = BMO_MATOPS_L2NORM(opt_objfn(x_vals,opt_data));
    fp_t Fx_p = BMO_MATOPS_L2NORM(opt_objfn(x_vals + lambda*direc,opt_data));
    fp_t direc_norm2 = BMO_MATOPS_DOT_PROD(direc,direc);

    fp_t term_2 = sigma_1 * (lambda*lambda) * direc_norm2;
    fp_t term_3 = eta_k * Fx;
    
    if (Fx_p <= Fx - term_2 + term_3) {
        return lambda;
    }

    // begin loop

    size_t iter = 0;
    uint_t max_iter = 10000;
    
    while (iter < max_iter) {
        ++iter;
        lambda *= beta; // lambda_i = beta^i;

        Fx_p = BMO_MATOPS_L2NORM( opt_objfn(x_vals + lambda*direc, opt_data) );
        term_2 = sigma_1 * (lambda*lambda) * direc_norm2;

        if (Fx_p <= Fx - term_2 + term_3) {
            break;
        }
    }

    //

    return lambda;
}

}

#endif
