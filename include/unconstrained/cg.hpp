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
cg(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data
);

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
cg(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t& settings
);

//
// internal

namespace internal
{

bool 
cg_impl(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t* settings_inp
);

// update function

inline
fp_t
cg_update(
    const ColVec_t& grad, 
    const ColVec_t& grad_p, 
    const ColVec_t& direc, 
    const uint_t iter, 
    const uint_t cg_method, 
    const fp_t cg_restart_threshold
)
{
    // threshold test
    fp_t ratio_value = std::abs( BMO_MATOPS_DOT_PROD(grad_p,grad) ) / BMO_MATOPS_DOT_PROD(grad_p,grad_p);

    if ( ratio_value > cg_restart_threshold ) {
        return 0.0;
    } else {
        fp_t beta = 1.0;

        switch (cg_method)
        {
            case 1: // Fletcher-Reeves (FR)
            {
                beta = BMO_MATOPS_DOT_PROD(grad_p,grad_p) / BMO_MATOPS_DOT_PROD(grad,grad);
                break;
            }

            case 2: // Polak-Ribiere (PR) + 
            {
                beta = BMO_MATOPS_DOT_PROD(grad_p, grad_p - grad) / BMO_MATOPS_DOT_PROD(grad,grad); // max(.,0.0) moved to end
                break;
            }

            case 3: // FR-PR hybrid, see eq. 5.48 in Nocedal and Wright
            {
                if (iter > 1) {
                    const fp_t beta_denom = BMO_MATOPS_DOT_PROD(grad, grad);
                    
                    const fp_t beta_FR = BMO_MATOPS_DOT_PROD(grad_p, grad_p) / beta_denom;
                    const fp_t beta_PR = BMO_MATOPS_DOT_PROD(grad_p, grad_p - grad) / beta_denom;
                    
                    if (beta_PR < - beta_FR) {
                        beta = -beta_FR;
                    } else if (std::abs(beta_PR) <= beta_FR) {
                        beta = beta_PR;
                    } else { // beta_PR > beta_FR
                        beta = beta_FR;
                    }
                } else {
                    // default to PR+
                    beta = BMO_MATOPS_DOT_PROD(grad_p,grad_p - grad) / BMO_MATOPS_DOT_PROD(grad,grad); // max(.,0.0) moved to end
                }
                break;
            }

            case 4: // Hestenes-Stiefel
            {
                beta = BMO_MATOPS_DOT_PROD(grad_p,grad_p - grad) / BMO_MATOPS_DOT_PROD(grad_p - grad,direc);
                break;
            }

            case 5: // Dai-Yuan
            {
                beta = BMO_MATOPS_DOT_PROD(grad_p,grad_p) / BMO_MATOPS_DOT_PROD(grad_p - grad,direc);
                break;
            }

            case 6: // Hager-Zhang
            {
                ColVec_t y = grad_p - grad;

                ColVec_t term_1 = y - 2*direc*(BMO_MATOPS_DOT_PROD(y,y) / BMO_MATOPS_DOT_PROD(y,direc));
                ColVec_t term_2 = grad_p / BMO_MATOPS_DOT_PROD(y,direc);

                beta = BMO_MATOPS_DOT_PROD(term_1,term_2);
                break;
            }
            
            default:
            {
                printf("error: unknown value for cg_method");
                break;
            }
        }

        //

        return std::max(beta, fp_t(0.0));
    }
}

}

#endif
