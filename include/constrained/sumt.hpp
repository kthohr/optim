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
 * Sequential unconstrained minimization technique (SUMT)
 */

#ifndef _optim_sumt_HPP
#define _optim_sumt_HPP

/**
 * @brief Sequential Unconstrained Minimization Technique
 *
 * @param init_out_vals a column vector of initial values, which will be replaced by the solution upon successful completion of the optimization algorithm.
 * @param opt_objfn the function to be minimized, taking three arguments:
 *   - \c vals_inp a vector of inputs;
 *   - \c grad_out a vector to store the gradient; and
 *   - \c opt_data additional data passed to the user-provided function.
 * @param opt_data additional data passed to the user-provided function.
 * @param constr_fn the constraint functions, in vector form, taking three arguments.
 * @param constr_data additional data passed to the constraints functions.
 *
 * @return a boolean value indicating successful completion of the optimization algorithm.
 */

bool 
sumt(Vec_t& init_out_vals, 
     std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
     void* opt_data,
     std::function<Vec_t (const Vec_t& vals_inp, Mat_t* jacob_out, void* constr_data)> constr_fn, 
     void* constr_data);

/**
 * @brief Sequential Unconstrained Minimization Technique
 *
 * @param init_out_vals a column vector of initial values, which will be replaced by the solution upon successful completion of the optimization algorithm.
 * @param opt_objfn the function to be minimized, taking three arguments:
 *   - \c vals_inp a vector of inputs;
 *   - \c grad_out a vector to store the gradient; and
 *   - \c opt_data additional data passed to the user-provided function.
 * @param opt_data additional data passed to the user-provided function.
 * @param constr_fn the constraint functions, in vector form, taking three arguments.
 * @param constr_data additional data passed to the constraints functions.
 * @param settings parameters controlling the optimization routine.
 *
 * @return a boolean value indicating successful completion of the optimization algorithm.
 */

bool 
sumt(Vec_t& init_out_vals, 
     std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
     void* opt_data,
     std::function<Vec_t (const Vec_t& vals_inp, Mat_t* jacob_out, void* constr_data)> constr_fn, 
     void* constr_data, 
     algo_settings_t& settings);

//
// internal

namespace internal
{

struct sumt_data_t {
    double c_pen;
};

inline
double
mt_sup_norm(const double a, 
            const double b, 
            const double c)
{
    return std::max( std::max(std::abs(a), std::abs(b)), std::abs(c) );
}

bool 
sumt_impl(Vec_t& init_out_vals, 
          std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
          void* opt_data,
          std::function<Vec_t (const Vec_t& vals_inp, Mat_t* jacob_out, void* constr_data)> constr_fn, 
          void* constr_data, 
          algo_settings_t* settings_inp);

}

#endif
