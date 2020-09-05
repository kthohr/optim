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
 * Conjugate Gradient method for non-linear optimization
 */

#ifndef _optim_cg_HPP
#define _optim_cg_HPP

/**
 * @brief The Nonlinear Conjugate Gradient (CG) Optimization Algorithm
 *
 * @param init_out_vals a column vector of initial values, which will be replaced by the solution upon successful completion of the optimization algorithm.
 * @param opt_objfn the function to be minimized, taking three arguments:
 *   - \c vals_inp a vector of inputs;
 *   - \c grad_out a vector to store the gradient; and
 *   - \c opt_data additional data passed to the user-provided function.
 * @param opt_data additional data passed to the user-provided function
 *
 * @return a boolean value indicating successful completion of the optimization algorithm.
 */

bool 
cg(Vec_t& init_out_vals, 
   std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
   void* opt_data);

/**
 * @brief The Nonlinear Conjugate Gradient (CG) Optimization Algorithm
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
cg(Vec_t& init_out_vals, 
   std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
   void* opt_data, 
   algo_settings_t& settings);

//
// internal

namespace internal
{

bool 
cg_impl(Vec_t& init_out_vals, 
        std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
        void* opt_data, 
        algo_settings_t* settings_inp);

// update function

inline
double
cg_update(const Vec_t& grad, 
          const Vec_t& grad_p, 
          const Vec_t& direc, 
          const uint_t iter, 
          const uint_t cg_method, 
          const double cg_restart_threshold)
{
    // threshold test
    double ratio_value = std::abs( OPTIM_MATOPS_DOT_PROD(grad_p,grad) ) / OPTIM_MATOPS_DOT_PROD(grad_p,grad_p);

    if ( ratio_value > cg_restart_threshold ) {
        return 0.0;
    } else {
        double beta = 1.0;

        switch (cg_method)
        {
            case 1: // Fletcher-Reeves (FR)
            {
                beta = OPTIM_MATOPS_DOT_PROD(grad_p,grad_p) / OPTIM_MATOPS_DOT_PROD(grad,grad);
                break;
            }

            case 2: // Polak-Ribiere (PR) + 
            {
                beta = OPTIM_MATOPS_DOT_PROD(grad_p, grad_p - grad) / OPTIM_MATOPS_DOT_PROD(grad,grad); // max(.,0.0) moved to end
                break;
            }

            case 3: // FR-PR hybrid, see eq. 5.48 in Nocedal and Wright
            {
                if (iter > 1) {
                    const double beta_denom = OPTIM_MATOPS_DOT_PROD(grad, grad);
                    
                    const double beta_FR = OPTIM_MATOPS_DOT_PROD(grad_p, grad_p) / beta_denom;
                    const double beta_PR = OPTIM_MATOPS_DOT_PROD(grad_p, grad_p - grad) / beta_denom;
                    
                    if (beta_PR < - beta_FR) {
                        beta = -beta_FR;
                    } else if (std::abs(beta_PR) <= beta_FR) {
                        beta = beta_PR;
                    } else { // beta_PR > beta_FR
                        beta = beta_FR;
                    }
                } else {
                    // default to PR+
                    beta = OPTIM_MATOPS_DOT_PROD(grad_p,grad_p - grad) / OPTIM_MATOPS_DOT_PROD(grad,grad); // max(.,0.0) moved to end
                }
                break;
            }

            case 4: // Hestenes-Stiefel
            {
                beta = OPTIM_MATOPS_DOT_PROD(grad_p,grad_p - grad) / OPTIM_MATOPS_DOT_PROD(grad_p - grad,direc);
                break;
            }

            case 5: // Dai-Yuan
            {
                beta = OPTIM_MATOPS_DOT_PROD(grad_p,grad_p) / OPTIM_MATOPS_DOT_PROD(grad_p - grad,direc);
                break;
            }

            case 6: // Hager-Zhang
            {
                Vec_t y = grad_p - grad;

                Vec_t term_1 = y - 2*direc*(OPTIM_MATOPS_DOT_PROD(y,y) / OPTIM_MATOPS_DOT_PROD(y,direc));
                Vec_t term_2 = grad_p / OPTIM_MATOPS_DOT_PROD(y,direc);

                beta = OPTIM_MATOPS_DOT_PROD(term_1,term_2);
                break;
            }
            
            default:
            {
                printf("error: unknown value for cg_method");
                break;
            }
        }

        //

        return std::max(beta, 0.0);
    }
}

}

#endif
