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
 * Error reporting
 */

#ifndef _optim_error_reporting_HPP
#define _optim_error_reporting_HPP

void error_reporting(
    ColVec_t& out_vals, 
    const ColVec_t& x_p, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data,
    bool& success, 
    const fp_t err, 
    const fp_t err_tol, 
    const size_t iter, 
    const size_t iter_max, 
    const int conv_failure_switch, 
    algo_settings_t* settings_inp
);

void error_reporting(ColVec_t& out_vals, 
    const ColVec_t& x_p, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data,
    bool& success, 
    const int conv_failure_switch, 
    algo_settings_t* settings_inp
);

void error_reporting(
    ColVec_t& out_vals, 
    const ColVec_t& x_p, 
    std::function<ColVec_t (const ColVec_t& vals_inp, void* opt_data)> opt_objfn, 
    void* opt_data,
    bool& success, 
    const fp_t err, 
    const fp_t err_tol, 
    const size_t iter, 
    const size_t iter_max, 
    const int conv_failure_switch, 
    algo_settings_t* settings_inp
);

//

void error_reporting(
    ColVec_t& out_vals, 
    const ColVec_t& x_p, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, Mat_t* hess_out, void* opt_data)> opt_objfn, 
    void* opt_data,
    bool& success, 
    const fp_t err, 
    const fp_t err_tol, 
    const size_t iter, 
    const size_t iter_max, 
    const int conv_failure_switch, 
    algo_settings_t* settings_inp
);

//

#include "error_reporting.ipp"

#endif
