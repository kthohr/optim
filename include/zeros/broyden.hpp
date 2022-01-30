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
 * Broyden's method for solving systems of nonlinear equations
 */

#ifndef _optim_broyden_HPP
#define _optim_broyden_HPP

// without jacobian

/**
 * @brief Broyden's method for solving systems of nonlinear equations, without Jacobian
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
broyden(
    ColVec_t& init_out_vals, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data
);

/**
 * @brief Broyden's method for solving systems of nonlinear equations, without Jacobian
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
broyden(
    ColVec_t& init_out_vals, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t& settings
);

//
// with jacobian

/**
 * @brief Broyden's method for solving systems of nonlinear equations, with Jacobian
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
broyden(
    ColVec_t& init_out_vals, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data,
    std::function<Mat_t (const ColVec_t& vals_inp, void* jacob_data)> jacob_objfn, 
    void* jacob_data
);

/**
 * @brief Broyden's method for solving systems of nonlinear equations, with Jacobian
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
broyden(ColVec_t& init_out_vals, 
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
broyden_impl(
    ColVec_t& init_out_vals, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t* settings_inp
);


bool
broyden_impl(
    ColVec_t& init_out_vals, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data,
    std::function<Mat_t (const ColVec_t& vals_inp, void* jacob_data)> jacob_objfn, 
    void* jacob_data, 
    algo_settings_t* settings_inp
);

}

#endif
