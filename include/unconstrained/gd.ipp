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
 * Gradient Descent (GD)
 */

#ifndef _optim_gd_IPP
#define _optim_gd_IPP

namespace internal
{

// update function

inline
ColVec_t
gd_update(
    const ColVec_t& vals_inp,
    const ColVec_t& grad,
    const ColVec_t& grad_p,
    const ColVec_t& direc,
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> box_objfn,
    void* opt_data,
    const size_t iter,
    gd_settings_t& gd_settings,
    ColVec_t& adam_vec_m,
    ColVec_t& adam_vec_v)
{
    ColVec_t direc_out; // direction

    if (gd_settings.step_decay) {
        if ((iter % gd_settings.step_decay_periods) == 0) {
            gd_settings.par_step_size *= gd_settings.step_decay_val;
        }
    }

    switch (gd_settings.method)
    {
        case 0: // basic
        {
            direc_out = gd_settings.par_step_size * grad_p;
            break;
        }

        case 1: // momentum
        {
            // direc_out = gd_settings.par_step_size * (gd_settings.par_momentum * direc + grad_p);
            direc_out = gd_settings.par_momentum * direc + gd_settings.par_step_size * grad_p;
            break;
        }

        case 2: // Nesterov accelerated gradient
        {
            ColVec_t NAG_grad( BMO_MATOPS_SIZE(vals_inp) );
            box_objfn(vals_inp - gd_settings.par_momentum * direc, &NAG_grad, opt_data);

            // direc_out = gd_settings.par_step_size * (gd_settings.par_momentum * direc + NAG_grad);
            direc_out = gd_settings.par_momentum * direc + gd_settings.par_step_size * NAG_grad;
            break;
        }

        case 3: // AdaGrad
        {
            adam_vec_v += BMO_MATOPS_POW(grad_p,2);

            direc_out = BMO_MATOPS_ARRAY_DIV_ARRAY( gd_settings.par_step_size * grad_p, BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_SQRT(adam_vec_v), gd_settings.par_ada_norm_term) );
            break;
        }

        case 4: // RMSProp
        {
            adam_vec_v = gd_settings.par_ada_rho * adam_vec_v + (fp_t(1.0) - gd_settings.par_ada_rho) * BMO_MATOPS_POW(grad_p,2);

            direc_out = BMO_MATOPS_ARRAY_DIV_ARRAY( gd_settings.par_step_size * grad_p, BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_SQRT(adam_vec_v), gd_settings.par_ada_norm_term) );
            break;
        }

        case 5: // Adadelta
        {
            if (iter == 1) {
                adam_vec_m = BMO_MATOPS_ARRAY_ADD_SCALAR(adam_vec_m, gd_settings.par_step_size);
            }

            adam_vec_v = gd_settings.par_ada_rho * adam_vec_v + (fp_t(1.0) - gd_settings.par_ada_rho) * BMO_MATOPS_POW(grad_p,2);

            ColVec_t grad_direc = BMO_MATOPS_ARRAY_DIV_ARRAY((BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_SQRT(adam_vec_m), gd_settings.par_ada_norm_term)), (BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_SQRT(adam_vec_v), gd_settings.par_ada_norm_term)));

            direc_out = BMO_MATOPS_HADAMARD_PROD(grad_p, grad_direc);

            adam_vec_m = gd_settings.par_ada_rho * adam_vec_m + (fp_t(1.0) - gd_settings.par_ada_rho) * BMO_MATOPS_POW(direc_out,2);
            break;
        }

        case 6: // Adam and AdaMax
        {
            adam_vec_m = gd_settings.par_adam_beta_1 * adam_vec_m + (fp_t(1.0) - gd_settings.par_adam_beta_1) * grad_p;

            if (gd_settings.ada_max) {
                adam_vec_v = BMO_MATOPS_MAX(gd_settings.par_adam_beta_2 * adam_vec_v, BMO_MATOPS_ABS(grad_p));

                fp_t adam_step_size = gd_settings.par_step_size / (fp_t(1.0) - std::pow(gd_settings.par_adam_beta_1,iter));

                direc_out = BMO_MATOPS_ARRAY_DIV_ARRAY( (adam_step_size * adam_vec_m), (BMO_MATOPS_ARRAY_ADD_SCALAR(adam_vec_v, gd_settings.par_ada_norm_term)) );
            } else {
                fp_t adam_step_size = gd_settings.par_step_size * std::sqrt(fp_t(1.0) - std::pow(gd_settings.par_adam_beta_2,iter)) \
                                     / (fp_t(1.0) - std::pow(gd_settings.par_adam_beta_1,iter));

                adam_vec_v = gd_settings.par_adam_beta_2 * adam_vec_v + (fp_t(1.0) - gd_settings.par_adam_beta_2) * BMO_MATOPS_POW(grad_p,2);

                direc_out = BMO_MATOPS_ARRAY_DIV_ARRAY( (adam_step_size * adam_vec_m), (BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_SQRT(adam_vec_v), gd_settings.par_ada_norm_term)) );
            }

            break;
        }

        case 7: // Nadam and NadaMax
        {
            adam_vec_m = gd_settings.par_adam_beta_1 * adam_vec_m + (fp_t(1.0) - gd_settings.par_adam_beta_1) * grad_p;

            if (gd_settings.ada_max) {
                adam_vec_v = BMO_MATOPS_MAX(gd_settings.par_adam_beta_2 * adam_vec_v, BMO_MATOPS_ABS(grad_p));

                ColVec_t m_hat = adam_vec_m / (fp_t(1.0) - std::pow(gd_settings.par_adam_beta_1,iter));
                ColVec_t grad_hat = grad_p / (fp_t(1.0) - std::pow(gd_settings.par_adam_beta_1,iter));

                direc_out = BMO_MATOPS_ARRAY_DIV_ARRAY( (gd_settings.par_step_size * ( gd_settings.par_adam_beta_1 * m_hat + (fp_t(1.0) - gd_settings.par_adam_beta_1) * grad_hat )) , (BMO_MATOPS_ARRAY_ADD_SCALAR(adam_vec_v, gd_settings.par_ada_norm_term)) );
            } else {
                adam_vec_v = gd_settings.par_adam_beta_2 * adam_vec_v + (fp_t(1.0) - gd_settings.par_adam_beta_2) * BMO_MATOPS_POW(grad_p,2);

                ColVec_t m_hat = adam_vec_m / (fp_t(1.0) - std::pow(gd_settings.par_adam_beta_1,iter));
                ColVec_t v_hat = adam_vec_v / (fp_t(1.0) - std::pow(gd_settings.par_adam_beta_2,iter));
                ColVec_t grad_hat = grad_p / (fp_t(1.0) - std::pow(gd_settings.par_adam_beta_1,iter));

                direc_out = BMO_MATOPS_ARRAY_DIV_ARRAY( (gd_settings.par_step_size * ( gd_settings.par_adam_beta_1 * m_hat + (fp_t(1.0) - gd_settings.par_adam_beta_1) * grad_hat )) , (BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_SQRT(v_hat), gd_settings.par_ada_norm_term)) );
            }

            break;
        }

        default:
        {
            printf("gd error: unknown value for gd_settings.method");
            break;
        }
    }

    return direc_out;
}

#endif

// gradient clipping

inline
void
gradient_clipping(
    ColVec_t& grad_, 
    const gd_settings_t& gd_settings
)
{
    fp_t grad_norm;
    
    if (gd_settings.clip_max_norm) {
        grad_norm = BMO_MATOPS_LINFNORM(grad_);
    } else if (gd_settings.clip_min_norm) {
        grad_norm = BMO_MATOPS_LMINFNORM(grad_);
    } else {
        grad_norm = BMO_MATOPS_LPNORM(grad_, gd_settings.clip_norm_type);
    }

    //

    if (grad_norm > gd_settings.clip_norm_bound) {
        if (std::isfinite(grad_norm)) {
            grad_ = gd_settings.clip_norm_bound * (grad_ / grad_norm);
        }
    }
}

}
