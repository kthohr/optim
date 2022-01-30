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
 * Moré and Thuente line search
 *
 * Based on MINPACK fortran code and Dianne P. O'Leary's Matlab-based translation of MINPACK
 */

#include "optim.hpp"

// [OPTIM_BEGIN]
optimlib_inline
optim::fp_t
optim::internal::line_search_mt(
    fp_t step, 
    ColVec_t& x, 
    ColVec_t& grad, 
    const ColVec_t& direc, 
    const fp_t* wolfe_cons_1_inp, 
    const fp_t* wolfe_cons_2_inp, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data
)
{
    const size_t iter_max = 100;

    const fp_t step_min = 0.0;
    const fp_t step_max = 10.0;
    const fp_t xtol = 1E-04;

    // Wolfe parameters
    const fp_t wolfe_cons_1 = (wolfe_cons_1_inp) ? *wolfe_cons_1_inp : 1E-03; // tolerance on the Armijo sufficient decrease condition; sometimes labelled 'mu'.
    const fp_t wolfe_cons_2 = (wolfe_cons_2_inp) ? *wolfe_cons_2_inp : 0.90;  // tolerance on the curvature condition; sometimes labelled 'eta'.
    
    //

    uint_t info = 0, infoc = 1;
    const fp_t extrap_delta = 4; // 'delta' on page 20

    ColVec_t x_0 = x;

    fp_t f_step = opt_objfn(x,&grad,opt_data); // q(0)

    fp_t dgrad_init = BMO_MATOPS_DOT_PROD(grad,direc);
    
    if (dgrad_init >= 0.0) {
        return step;
    }

    fp_t dgrad = dgrad_init;

    //

    size_t iter = 0;

    bool bracket = false, stage_1 = true;

    fp_t f_init = f_step, dgrad_test = wolfe_cons_1*dgrad_init;
    fp_t width = step_max - step_min, width_old = 2*width;
    
    fp_t st_best = 0.0, f_best = f_init, dgrad_best = dgrad_init;
    fp_t st_other = 0.0, f_other = f_init, dgrad_other = dgrad_init;

    while (1) {
        ++iter;

        fp_t st_min, st_max;

        if (bracket) {
            st_min = std::min(st_best,st_other);
            st_max = std::max(st_best,st_other);
        } else {
            st_min = st_best;
            st_max = step + extrap_delta*(step - st_best);
        }

        step = std::min(std::max(step,step_min),step_max);

        if ( (bracket && (step <= st_min || step >= st_max)) \
                || iter >= iter_max-1 || infoc == 0 || (bracket && st_max-st_min <= xtol*st_max) ) {
            step = st_best;
        }

        //

        x = x_0 + step * direc;
        f_step = opt_objfn(x,&grad,opt_data);

        dgrad = BMO_MATOPS_DOT_PROD(grad,direc);
        fp_t armijo_check_val = f_init + step*dgrad_test;

        // check stop conditions

        if ((bracket && (step <= st_min || step >= st_max)) || infoc == 0) {
            info = 6;
        }
        if (step == step_max && f_step <= armijo_check_val && dgrad <= dgrad_test) {
            info = 5;
        }
        if (step == step_min && (f_step > armijo_check_val || dgrad >= dgrad_test)) {
            info = 4;
        }
        if (iter >= iter_max) {
            info = 3;
        }
        if (bracket && st_max-st_min <= xtol*st_max) {
            info = 2;
        }

        if (f_step <= armijo_check_val && std::abs(dgrad) <= wolfe_cons_2*(-dgrad_init))
        {   // strong Wolfe conditions
            info = 1;
        }

        if (info != 0) {
            return step;
        }

        //

        if (stage_1 && f_step <= armijo_check_val && dgrad >= std::min(wolfe_cons_1,wolfe_cons_2)*dgrad_init) {
            stage_1 = false;
        }

        if (stage_1 && f_step <= f_best && f_step > armijo_check_val) {
            fp_t f_mod  = f_step - step*dgrad_test;
            fp_t f_best_mod = f_best - st_best*dgrad_test;
            fp_t f_other_mod = f_other - st_other*dgrad_test;

            fp_t dgrad_mod  = dgrad - dgrad_test;
            fp_t dgrad_best_mod = dgrad_best - dgrad_test;
            fp_t dgrad_other_mod = dgrad_other - dgrad_test;

            infoc = mt_step(st_best,f_best_mod,dgrad_best_mod,st_other,f_other_mod,dgrad_other_mod,step,f_mod,dgrad_mod,bracket,st_min,st_max);
            
            //

            f_best = f_best_mod + st_best*dgrad_test;
            f_other = f_other_mod + st_other*dgrad_test;

            dgrad_best = dgrad_best_mod + dgrad_test;
            dgrad_other = dgrad_other_mod + dgrad_test;
        } else {
            infoc = mt_step(st_best,f_best,dgrad_best,st_other,f_other,dgrad_other,step,f_step,dgrad,bracket,st_min,st_max);
        }

        //

        if (bracket) {
            if (std::abs(st_other - st_best) >= 0.66*width_old) {
                step = st_best + 0.5*(st_other - st_best);
            }

            width_old = width;
            width = std::abs(st_other - st_best);
        }
    }

    //

    return step;
}

//
// update the 'interval of uncertainty'

optimlib_inline
optim::uint_t
optim::internal::mt_step(
    fp_t& st_best, 
    fp_t& f_best, 
    fp_t& d_best, 
    fp_t& st_other, 
    fp_t& f_other, 
    fp_t& d_other, 
    fp_t& step, 
    fp_t& f_step, 
    fp_t& d_step, 
    bool& bracket, 
    fp_t step_min, 
    fp_t step_max
)
{
    bool bound = false;
    uint_t info = 0;
    fp_t sgnd = d_step*(d_best / std::abs(d_best));

    fp_t theta,s,gamma, p,q,r, step_c,step_q,step_f;

    if (f_step > f_best) {
        info = 1;
        bound = true;

        theta = 3*(f_best - f_step)/(step - st_best) + d_best + d_step;
        s = mt_sup_norm(theta,d_best,d_step); // sup norm

        gamma = s*std::sqrt(std::pow(theta/s,2) - (d_best/s)*(d_step/s));
        if (step < st_best) {
            gamma = -gamma;
        }

        p = (gamma - d_best) + theta;
        q = ((gamma - d_best) + gamma) + d_step;
        r = p/q;

        step_c = st_best + r*(step - st_best);
        step_q = st_best + ((d_best / ((f_best - f_step)/(step - st_best) + d_best)) / 2.0)*(step - st_best);

        if (std::abs(step_c - st_best) < std::abs(step_q - st_best)) {
            step_f = step_c;
        } else {
            step_f = step_c + (step_q - step_c)/2;
        }

        bracket = true;
    } else if (sgnd < fp_t(0.0)) {
        info = 2;
        bound = false;
     
        theta = 3*(f_best - f_step)/(step - st_best) + d_best + d_step;
        s = mt_sup_norm(theta,d_best,d_step); // sup norm

        gamma = s * std::sqrt(std::pow(theta/s,fp_t(2)) - (d_best/s)*(d_step/s));
        if (step > st_best) {
            gamma = -gamma;
        }

        p = (gamma - d_step) + theta;
        q = ((gamma - d_step) + gamma) + d_best;
        r = p/q;

        step_c = step + r*(st_best - step);
        step_q = step + (d_step/(d_step-d_best))*(st_best - step);

        if (std::abs(step_c-step) > std::abs(step_q-step)) {
            step_f = step_c;
        } else {
            step_f = step_q;
        }

        bracket = true;
    } else if (std::abs(d_step) < std::abs(d_best)) {
        info = 3;
        bound = true;

        theta = 3*(f_best - f_step)/(step - st_best) + d_best + d_step;
        s = mt_sup_norm(theta,d_best,d_step); // sup norm

        gamma = s * std::sqrt(std::max(fp_t(0.0), std::pow(theta/s,fp_t(2)) - (d_best/s)*(d_step/s)));
        if (step > st_best) {
            gamma = -gamma;
        }

        p = (gamma - d_step) + theta;
        q = (gamma + (d_best - d_step)) + gamma;
        r = p/q;

        if (r < fp_t(0.0) && gamma != fp_t(0.0)) {
            step_c = step + r*(st_best - step);
        } else if (step > st_best) {
            step_c = step_max;
        } else {
            step_c = step_min;
        }

        step_q = step + (d_step/(d_step-d_best))*(st_best - step);

        if (bracket) {
            if (std::abs(step-step_c) < std::abs(step-step_q)) {
                step_f = step_c;
            } else {
                step_f = step_q;
            }
        } else {
            if (std::abs(step-step_c) > std::abs(step-step_q)) {
                step_f = step_c;
            } else {
                step_f = step_q;
            }
        }
    } else {
        info = 4;
        bound = false;

        if (bracket) {
            theta = 3*(f_step - f_other)/(st_other - step) + d_other + d_step;
            s = mt_sup_norm(theta,d_other,d_step);

            gamma = s*std::sqrt(std::pow(theta/s,fp_t(2)) - (d_other/s)*(d_step/s));
            if (step > st_other) {
                gamma = -gamma;
            }

            p = (gamma - d_step) + theta;
            q = ((gamma - d_step) + gamma) + d_other;
            r = p/q;

            step_c = step + r*(st_other - step);
            step_f = step_c;
        }  else if (step > st_best) {
            step_f = step_max;
        } else {
            step_f = step_min;
        }
    }

    /*
     * Update the interval of uncertainty.
     */

    if (f_step > f_best) {
        st_other = step;
        f_other = f_step;
        d_other = d_step;
    } else {
        if (sgnd < fp_t(0.0)) {
            st_other = st_best;
            f_other = f_best;
            d_other = d_best;
        }

        st_best = step;
        f_best = f_step;
        d_best = d_step;
    }

    /*
     * Compute the new step and safeguard it.
     */

    step_f = std::max(step_min, std::min(step_max,step_f));
    step = step_f;
    
    if (bracket && bound) {
        if (st_other > st_best) {
            step = std::min(st_best + fp_t(0.66) * (st_other - st_best), step);
        } else {
            step = std::max(st_best + fp_t(0.66) * (st_other - st_best), step);
        }
    }

    //

    return info;
}
