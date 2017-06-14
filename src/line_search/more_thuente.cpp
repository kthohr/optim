/*################################################################################
  ##
  ##   Copyright (C) 2016-2017 Keith O'Hara
  ##
  ##   This file is part of the OptimLib C++ library.
  ##
  ##   OptimLib is free software: you can redistribute it and/or modify
  ##   it under the terms of the GNU General Public License as published by
  ##   the Free Software Foundation, either version 2 of the License, or
  ##   (at your option) any later version.
  ##
  ##   OptimLib is distributed in the hope that it will be useful,
  ##   but WITHOUT ANY WARRANTY; without even the implied warranty of
  ##   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  ##   GNU General Public License for more details.
  ##
  ################################################################################*/

/*
 * Moré and Thuente line search
 *
 * Based on MINPACK fortran code and Dianne P. O'Leary's Matlab translation of MINPACK
 *
 * Keith O'Hara
 * 01/03/2017
 *
 * This version:
 * 01/11/2017
 */

#include "optim.hpp"

double optim::line_search_mt(double step, arma::vec& x, arma::vec& grad, const arma::vec& direc, double* wolfe_cons_1_inp, double* wolfe_cons_2_inp, std::function<double (const arma::vec& vals_inp, arma::vec* grad, void* opt_data)> opt_objfn, void* opt_data)
{
    int iter_max = 100;

    double step_min = 0.0;
    double step_max = 10.0;
    double xtol = 1E-04;

    // Wolfe parameters
    double wolfe_cons_1 = (wolfe_cons_1_inp) ? *wolfe_cons_1_inp : 1E-03; // tolerence on the Armijo sufficient decrease condition; sometimes labelled 'mu'.
    double wolfe_cons_2 = (wolfe_cons_2_inp) ? *wolfe_cons_2_inp : 0.90;  // tolerence on the curvature condition; sometimes labelled 'eta'.
    //
    int info = 0, infoc = 1;
    double extrap_delta = 4; // page 20 'delta'

    arma::vec x_0 = x;

    double f_step = opt_objfn(x,&grad,opt_data); // q(0)

    double dgrad_init = arma::dot(grad,direc);
    if (dgrad_init >= 0.0) {
        return step;
    }
    double dgrad = dgrad_init;
    //
    bool bracket = false, stage_1 = true;
    int iter = 0;

    double f_init = f_step, dgrad_test = wolfe_cons_1*dgrad_init;
    double width = step_max - step_min, width_old = 2*width;
    
    double st_best = 0.0, f_best = f_init, dgrad_best = dgrad_init;
    double st_other = 0.0, f_other = f_init, dgrad_other = dgrad_init;

    double st_min, st_max;
    double armijo_check, f_mod, f_best_mod, f_other_mod, dgrad_mod, dgrad_best_mod, dgrad_other_mod;

    while (1) {
        iter++;

        if (bracket) {
            st_min = std::min(st_best,st_other);
            st_max = std::max(st_best,st_other);
        } else {
            st_min = st_best;
            st_max = step + extrap_delta*(step - st_best);
        }

        step = std::max(step,step_min);
        step = std::min(step,step_max);

        if ((bracket && (step <= st_min || step >= st_max)) || iter >= iter_max-1 || infoc == 0 || (bracket && st_max-st_min <= xtol*st_max)) {
            step = st_best;
        }

        x = x_0 + step * direc;
        f_step = opt_objfn(x,&grad,opt_data);

        dgrad = arma::dot(grad,direc);
        armijo_check = f_init + step*dgrad_test;
        //
        if ((bracket && (step <= st_min || step >= st_max)) || infoc == 0) {
            info = 6;
        }
        if (step == step_max && f_step <= armijo_check && dgrad <= dgrad_test) {
            info = 5;
        }
        if (step == step_min && (f_step > armijo_check || dgrad >= dgrad_test)) {
            info = 4;
        }
        if (iter >= iter_max) {
            info = 3;
        }
        if (bracket && st_max-st_min <= xtol*st_max) {
            info = 2;
        }
        if (f_step <= armijo_check && std::abs(dgrad) <= wolfe_cons_2*(-dgrad_init)) { // strong Wolfe conditions
            info = 1;
        }

        if (info != 0) {
            return step;
        }
        //
        if (stage_1 && f_step <= armijo_check && dgrad >= std::min(wolfe_cons_1,wolfe_cons_2)*dgrad_init) {
            stage_1 = false;
        }

        if (stage_1 && f_step <= f_best && f_step > armijo_check) {
            f_mod  = f_step - step*dgrad_test;
            f_best_mod = f_best - st_best*dgrad_test;
            f_other_mod = f_other - st_other*dgrad_test;

            dgrad_mod  = dgrad - dgrad_test;
            dgrad_best_mod = dgrad_best - dgrad_test;
            dgrad_other_mod = dgrad_other - dgrad_test;

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

// update 'interval of uncertainty'
int optim::mt_step(double& st_best, double& f_best, double& d_best, double& st_other, double& f_other, double& d_other, double& step, double& f_step, double& d_step, bool& bracket, double step_min, double step_max)
{
    bool bound = false;
    int info = 0;
    double sgnd = d_step*(d_best / std::abs(d_best));

    double theta,s,gamma, p,q,r, step_c,step_q,step_f;

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
        step_q = st_best + ((d_best / ((f_best - f_step)/(step - st_best) + d_best)) / 2)*(step - st_best);

        if (std::abs(step_c - st_best) < std::abs(step_q - st_best)) {
            step_f = step_c;
        } else {
            step_f = step_c + (step_q - step_c)/2;
        }

        bracket = true;
    } else if (sgnd < 0.0) {
        info = 2;
        bound = false;
     
        theta = 3*(f_best - f_step)/(step - st_best) + d_best + d_step;
        s = mt_sup_norm(theta,d_best,d_step); // sup norm

        gamma = s*std::sqrt(std::pow(theta/s,2) - (d_best/s)*(d_step/s));
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

        gamma = s*std::sqrt(std::max(0.0,std::pow(theta/s,2) - (d_best/s)*(d_step/s)));
        if (step > st_best) {
            gamma = -gamma;
        }

        p = (gamma - d_step) + theta;
        q = (gamma + (d_best - d_step)) + gamma;
        r = p/q;

        if (r < 0.0 && gamma != 0.0) {
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

            gamma = s*std::sqrt(std::pow(theta/s,2) - (d_other/s)*(d_step/s));
            if (step > st_other) {
                gamma = -gamma;
            }

            p = (gamma - d_step) + theta;
            q = ((gamma - d_step) + gamma) + d_other;
            r = p/q;

            step_c = step + r*(st_other - step);
            step_f = step_c;
        } else if (step > st_best) {
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
        if (sgnd < 0.0) {
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
    step_f = std::min(step_max,step_f);
    step_f = std::max(step_min,step_f);
    step = step_f;
    
    if (bracket && bound) {
        if (st_other > st_best) {
            step = std::min(st_best + 0.66*(st_other - st_best),step);
        } else {
            step = std::max(st_best + 0.66*(st_other - st_best),step);
        }
    }
    //
    return info;
}

double optim::mt_sup_norm(double a, double b, double c)
{
    double ret = std::max(std::abs(a), std::abs(b));
    ret = std::max(ret, std::abs(c));

    return ret;
}
