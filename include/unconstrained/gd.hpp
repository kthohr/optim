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
 * Gradient Descent (GD)
 */

#ifndef _optim_gd_HPP
#define _optim_gd_HPP

bool gd_basic_int(Vec_t& init_out_vals, 
                  std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
                  void* opt_data, 
                  algo_settings_t* settings_inp);

bool gd(Vec_t& init_out_vals, 
        std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
        void* opt_data);

bool gd(Vec_t& init_out_vals, 
        std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
        void* opt_data, 
        algo_settings_t& settings);

// internal update function

inline
Vec_t
gd_update(const Vec_t& vals_inp, 
          const Vec_t& grad, 
          const Vec_t& grad_p, 
          const Vec_t& direc,
          std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> box_objfn, 
          void* opt_data,
          const uint_t iter, 
          const uint_t gd_method, 
          gd_settings_t& gd_settings,
          Vec_t& adam_vec_m, 
          Vec_t& adam_vec_v)
{
    Vec_t direc_out; // direction

    if (gd_settings.step_decay) {
        if ((iter % gd_settings.step_decay_periods) == 0) {
            gd_settings.step_size *= gd_settings.step_decay_val;
        }
    }

    switch (gd_method)
    {
        case 0: // basic
        {
            direc_out = gd_settings.step_size * grad_p;
            break;
        }

        case 1: // momentum
        {
            // direc_out = gd_settings.step_size * (gd_settings.momentum_par * direc + grad_p);
            direc_out = gd_settings.momentum_par * direc + gd_settings.step_size * grad_p;
            break;
        }

        case 2: // Nesterov accelerated gradient
        {
            Vec_t NAG_grad( OPTIM_MATOPS_SIZE(vals_inp) );
            box_objfn(vals_inp - gd_settings.momentum_par * direc, &NAG_grad, opt_data);

            // direc_out = gd_settings.step_size * (gd_settings.momentum_par * direc + NAG_grad);
            direc_out = gd_settings.momentum_par * direc + gd_settings.step_size * NAG_grad;
            break;
        }

        case 3: // AdaGrad
        {
            adam_vec_v += OPTIM_MATOPS_POW(grad_p,2);

            direc_out = OPTIM_MATOPS_ARRAY_DIV_ARRAY( gd_settings.step_size * grad_p, OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_SQRT(adam_vec_v), gd_settings.norm_term) );
            break;
        }

        case 4: // RMSProp
        {
            adam_vec_v = gd_settings.ada_rho * adam_vec_v + (1.0 - gd_settings.ada_rho) * OPTIM_MATOPS_POW(grad_p,2);

            direc_out = OPTIM_MATOPS_ARRAY_DIV_ARRAY( gd_settings.step_size * grad_p, OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_SQRT(adam_vec_v), gd_settings.norm_term) );
            break;
        }

        case 5: // Adadelta
        {
            if (iter == 1) {
                adam_vec_m = OPTIM_MATOPS_ARRAY_ADD_SCALAR(adam_vec_m, gd_settings.step_size);
            }

            adam_vec_v = gd_settings.ada_rho * adam_vec_v + (1.0 - gd_settings.ada_rho) * OPTIM_MATOPS_POW(grad_p,2);

            Vec_t grad_direc = OPTIM_MATOPS_ARRAY_DIV_ARRAY((OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_SQRT(adam_vec_m), gd_settings.norm_term)), (OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_SQRT(adam_vec_v), gd_settings.norm_term)));

            direc_out = OPTIM_MATOPS_HADAMARD_PROD(grad_p, grad_direc);

            adam_vec_m = gd_settings.ada_rho * adam_vec_m + (1.0 - gd_settings.ada_rho) * OPTIM_MATOPS_POW(direc_out,2);
            break;
        }

        case 6: // Adam and AdaMax
        {
            adam_vec_m = gd_settings.adam_beta_1 * adam_vec_m + (1.0 - gd_settings.adam_beta_1) * grad_p;

            if (gd_settings.ada_max) {
                adam_vec_v = OPTIM_MATOPS_MAX(gd_settings.adam_beta_2 * adam_vec_v, OPTIM_MATOPS_ABS(grad_p));

                double adam_step_size = gd_settings.step_size / (1.0 - std::pow(gd_settings.adam_beta_1,iter));

                direc_out = OPTIM_MATOPS_ARRAY_DIV_ARRAY( (adam_step_size * adam_vec_m), (OPTIM_MATOPS_ARRAY_ADD_SCALAR(adam_vec_v, gd_settings.norm_term)) );
            } else {
                double adam_step_size = gd_settings.step_size * std::sqrt(1.0 - std::pow(gd_settings.adam_beta_2,iter)) \
                                     / (1.0 - std::pow(gd_settings.adam_beta_1,iter));

                adam_vec_v = gd_settings.adam_beta_2 * adam_vec_v + (1.0 - gd_settings.adam_beta_2) * OPTIM_MATOPS_POW(grad_p,2);

                direc_out = OPTIM_MATOPS_ARRAY_DIV_ARRAY( (adam_step_size * adam_vec_m), (OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_SQRT(adam_vec_v), gd_settings.norm_term)) );
            }

            break;
        }

        case 7: // Nadam and NadaMax
        {
            adam_vec_m = gd_settings.adam_beta_1 * adam_vec_m + (1.0 - gd_settings.adam_beta_1) * grad_p;

            if (gd_settings.ada_max) {
                adam_vec_v = OPTIM_MATOPS_MAX(gd_settings.adam_beta_2 * adam_vec_v, OPTIM_MATOPS_ABS(grad_p));

                Vec_t m_hat = adam_vec_m / (1.0 - std::pow(gd_settings.adam_beta_1,iter));
                Vec_t grad_hat = grad_p / (1.0 - std::pow(gd_settings.adam_beta_1,iter));

                direc_out = OPTIM_MATOPS_ARRAY_DIV_ARRAY( (gd_settings.step_size * ( gd_settings.adam_beta_1 * m_hat + (1.0 - gd_settings.adam_beta_1) * grad_hat )) , (OPTIM_MATOPS_ARRAY_ADD_SCALAR(adam_vec_v, gd_settings.norm_term)) );
            } else {
                adam_vec_v = gd_settings.adam_beta_2 * adam_vec_v + (1.0 - gd_settings.adam_beta_2) * OPTIM_MATOPS_POW(grad_p,2);

                Vec_t m_hat = adam_vec_m / (1.0 - std::pow(gd_settings.adam_beta_1,iter));
                Vec_t v_hat = adam_vec_v / (1.0 - std::pow(gd_settings.adam_beta_2,iter));
                Vec_t grad_hat = grad_p / (1.0 - std::pow(gd_settings.adam_beta_1,iter));

                direc_out = OPTIM_MATOPS_ARRAY_DIV_ARRAY( (gd_settings.step_size * ( gd_settings.adam_beta_1 * m_hat + (1.0 - gd_settings.adam_beta_1) * grad_hat )) , (OPTIM_MATOPS_ARRAY_ADD_SCALAR(OPTIM_MATOPS_SQRT(v_hat), gd_settings.norm_term)) );
            }

            break;
        }

        default:
        {
            printf("error: unknown value for gd_method");
            break;
        }
    }

    return direc_out;
}

#endif

// gradient clipping

inline
void
gradient_clipping(Vec_t& grad_, const gd_settings_t& settings_)
{

    double grad_norm;
    
    if (settings_.clipping.max_norm) {
        grad_norm = OPTIM_MATOPS_LINFNORM(grad_);
    } else if (settings_.clipping.min_norm) {
        grad_norm = OPTIM_MATOPS_LMINFNORM(grad_);
    } else {
        grad_norm = OPTIM_MATOPS_LPNORM(grad_, settings_.clipping.norm_type);
    }

    //

    if (grad_norm > settings_.clipping.norm_bound) {
        if (std::isfinite(grad_norm)) {
            grad_ = settings_.clipping.norm_bound * (grad_ / grad_norm);
        }
    }
}
