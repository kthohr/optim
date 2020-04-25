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

#include "optim.hpp"

// [OPTIM_BEGIN]
struct sumt_struct {
    double c_pen;
};

optimlib_inline
bool
optim::sumt_int(Vec_t& init_out_vals, 
                std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
                void* opt_data,
                std::function<Vec_t (const Vec_t& vals_inp, Mat_t* jacob_out, void* constr_data)> constr_fn, 
                void* constr_data, 
                algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.

    bool success = false;

    //
    // SUMT settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int conv_failure_switch = settings.conv_failure_switch;
    const int iter_max = settings.iter_max;
    const double err_tol = settings.err_tol;

    const double par_eta = settings.sumt_par_eta; // growth of penalty parameter

    // lambda function that combines the objective function with the constraints

    std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* sumt_data)> sumt_objfn \
    = [opt_objfn, opt_data, constr_fn, constr_data] (const Vec_t& vals_inp, Vec_t* grad_out, void* sumt_data) \
    -> double
    {
        sumt_struct *d = reinterpret_cast<sumt_struct*>(sumt_data);
        double c_pen = d->c_pen;
        
        const size_t n_vals = OPTIM_MATOPS_SIZE(vals_inp);

        Vec_t grad_obj(n_vals);
        Mat_t jacob_constr;

        //

        double ret = 1E08;

        Vec_t constr_vals = constr_fn(vals_inp, &jacob_constr, constr_data);
        Vec_t tmp_vec = constr_vals;

        reset_negative_values(tmp_vec, constr_vals);
        reset_negative_rows(tmp_vec, jacob_constr);

        //

        double constr_valsq = OPTIM_MATOPS_DOT_PROD(constr_vals,constr_vals);

        if (constr_valsq > 0) {
            ret = opt_objfn(vals_inp,&grad_obj,opt_data) + c_pen*(constr_valsq / 2.0);

            if (grad_out) {
                *grad_out = grad_obj + c_pen * OPTIM_MATOPS_TRANSPOSE( OPTIM_MATOPS_COLWISE_SUM(jacob_constr) );
            }
        } else {
            ret = opt_objfn(vals_inp, &grad_obj, opt_data);

            if (grad_out) {
                *grad_out = grad_obj;
            }
        }

        //

        return ret;
    };

    //
    // initialization
    
    Vec_t x = init_out_vals;

    sumt_struct sumt_data;
    sumt_data.c_pen = 1.0;

    Vec_t x_p = x;

    //
    // begin loop
    
    int iter = 0;
    double err = 2*err_tol;

    while (err > err_tol && iter < iter_max) {
        ++iter;

        //

        bfgs(x_p, sumt_objfn, &sumt_data, settings);

        if (iter % 10 == 0) {
            err = OPTIM_MATOPS_L2NORM(x_p - x);
        }
        
        //

        sumt_data.c_pen = par_eta * sumt_data.c_pen; // increase penalization parameter value
        x = x_p;
    }

    //

    error_reporting(init_out_vals, x_p, opt_objfn, opt_data, success, err, err_tol, iter, iter_max, conv_failure_switch, settings_inp);

    return success;
}

optimlib_inline
bool
optim::sumt(Vec_t& init_out_vals, 
            std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
            void* opt_data,
            std::function<Vec_t (const Vec_t& vals_inp, Mat_t* jacob_out, void* constr_data)> constr_fn, 
            void* constr_data)
{
    return sumt_int(init_out_vals, opt_objfn, opt_data, constr_fn, constr_data, nullptr);
}

optimlib_inline
bool
optim::sumt(Vec_t& init_out_vals, 
            std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
            void* opt_data,
            std::function<Vec_t (const Vec_t& vals_inp, Mat_t* jacob_out, void* constr_data)> constr_fn, 
            void* constr_data, 
            algo_settings_t& settings)
{
    return sumt_int(init_out_vals, opt_objfn, opt_data, constr_fn, constr_data, &settings);
}
