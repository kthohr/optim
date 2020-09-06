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
 * Error reporting
 */

inline
void
error_reporting(Vec_t& out_vals, 
                const Vec_t& x_p, 
                std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
                void* opt_data, 
                bool& success, 
                const double err, 
                const double err_tol, 
                const size_t iter, 
                const size_t iter_max, 
                const int conv_failure_switch, 
                algo_settings_t* settings_inp)
{
    success = false;

    if (conv_failure_switch == 0) {
        out_vals = x_p;

        if (err <= err_tol && iter <= iter_max) {
            success = true;
        }
    } else if (conv_failure_switch == 1) {
        out_vals = x_p;

        if (err <= err_tol && iter <= iter_max) {
            success = true;
        } else if (err > err_tol && iter < iter_max) {
            printf("optim failure: iter_max not reached but algorithm stopped.\n");
            printf("optim failure: returned best guess.\n");
            
            std::cout << "iterations: " << iter << ". error: " << err << std::endl;
        } else {
            printf("optim failure: iter_max reached before convergence could be achieved.\n");
            printf("optim failure: returned best guess.\n");
            
            std::cout << "iterations: " << iter << ". error: " << err << std::endl;
        }
    } else if (conv_failure_switch == 2) {
        if (err <= err_tol && iter <= iter_max) {
            out_vals = x_p;
            success = true;
        } else {
            printf("optim failure: iter_max reached before convergence could be achieved.\n");
            printf("optim failure: best guess:\n");

            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE_IN_PLACE(x_p) << "\n";
            std::cout << "iterations: " << iter << ". error: " << err << std::endl;
        }
    } else {
        printf("optim failure: unrecognized conv_failure_switch value.\n");
        success = false;
    }
    
    if (settings_inp) {
        settings_inp->opt_fn_value     = opt_objfn(x_p,nullptr,opt_data);
        settings_inp->opt_iter         = iter;
        settings_inp->opt_error_value  = err;
    }
}

inline
void
error_reporting(Vec_t& out_vals, 
                const Vec_t& x_p, 
                std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> opt_objfn, 
                void* opt_data,
                bool& success, 
                const int conv_failure_switch, 
                algo_settings_t* settings_inp)
{
    if (conv_failure_switch == 0 || conv_failure_switch == 1) {
        out_vals = x_p;
    } else if (conv_failure_switch == 2) {
        if (success) {
            out_vals = x_p;
        }
    } else {
        printf("optim failure: unrecognized conv_failure_switch value.\n");
        success = false;
    }
    
    if (settings_inp) {
        settings_inp->opt_fn_value = opt_objfn(x_p, nullptr, opt_data);
    }
}

inline
void
error_reporting(Vec_t& out_vals, 
                const Vec_t& x_p, 
                std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                void* opt_data,
                bool& success, 
                const double err, 
                const double err_tol, 
                const size_t iter, 
                const size_t iter_max, 
                const int conv_failure_switch, 
                algo_settings_t* settings_inp)
{
    success = false;

    if (conv_failure_switch == 0) {
        out_vals = x_p;

        if (err <= err_tol && iter <= iter_max) {
            success = true;
        }
    } else if (conv_failure_switch == 1) {
        out_vals = x_p;

        if (err <= err_tol && iter <= iter_max) {
            success = true;
        } else if (err > err_tol && iter < iter_max) {
            printf("optim failure: iter_max not reached but algorithm stopped.\n");
            printf("optim failure: returned best guess.\n");
            
            std::cout << "iterations: " << iter << ". error: " << err << std::endl;
        } else {
            printf("optim failure: iter_max reached before convergence could be achieved.\n");
            printf("optim failure: returned best guess.\n");
            
            std::cout << "error: " << err << std::endl;
        }
    } else if (conv_failure_switch == 2) {
        if (err <= err_tol && iter <= iter_max) {
            out_vals = x_p;
            success = true;
        } else if (err > err_tol && iter < iter_max) {
            printf("optim failure: iter_max not reached but algorithm stopped.\n");
            printf("optim failure: best guess:\n");

            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE_IN_PLACE(x_p) << "\n";
            std::cout << "error: " << err << std::endl;
        } else {
            printf("optim failure: iter_max reached before convergence could be achieved.\n");
            printf("optim failure: best guess:\n");

            OPTIM_MATOPS_COUT << OPTIM_MATOPS_TRANSPOSE_IN_PLACE(x_p) << "\n";
            std::cout << "error: " << err << std::endl;
        }
    } else {
        printf("optim failure: unrecognized conv_failure_switch value.\n");
        success = false;
    }

    if (settings_inp) {
        settings_inp->opt_root_fn_values = opt_objfn(x_p, opt_data);
        settings_inp->opt_iter           = iter;
        settings_inp->opt_error_value    = err;
    }
}

//

inline
void
error_reporting(Vec_t& out_vals, 
                const Vec_t& x_p, 
                std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, Mat_t* hess_out, void* opt_data)> opt_objfn, 
                void* opt_data,
                bool& success, 
                const double err, 
                const double err_tol, 
                const size_t iter, 
                const size_t iter_max, 
                const int conv_failure_switch, 
                algo_settings_t* settings_inp)
{
    std::function<double (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data)> lam_objfn \
    = [opt_objfn] (const Vec_t& vals_inp, Vec_t* grad_out, void* opt_data) 
    -> double 
    {
        return opt_objfn(vals_inp,grad_out,nullptr,opt_data);
    };

    //

    error_reporting(out_vals, x_p, lam_objfn, opt_data, success, err, err_tol, iter, iter_max, conv_failure_switch, settings_inp);
}