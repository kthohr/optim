/*################################################################################
  ##
  ##   Copyright (C) 2016-2018 Keith O'Hara
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

//
// coverage tests for error_reporting
//

#include "optim.hpp"

double optim_simple_fn_1(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)
{
    return 1.0;
}

arma::vec optim_simple_fn_2(const arma::vec& vals_inp, void* opt_data)
{
    int n = vals_inp.n_elem;
    return arma::zeros(n,1);
}

int main()
{
    
    arma::vec out_vals = arma::ones(2,1);
    arma::vec x_p = arma::ones(2,1);

    bool success = false;

    double err_1 = 0.5;
    double err_2 = 1.5;
    double err_tol = 1.0;

    int iter_1 = 1;
    int iter_2 = 3;
    int iter_max = 2;

    optim::algo_settings_t settings;

    std::cout << "\n     ***** Begin ERROR_REPORTING tests. *****     \n" << std::endl;

    //
    // error_reporting_1

    int conv_failure_switch = 0;

    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,err_1,err_tol,iter_1,iter_max,conv_failure_switch,&settings);
    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,err_2,err_tol,iter_2,iter_max,conv_failure_switch,&settings);

    conv_failure_switch = 1;

    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,err_1,err_tol,iter_1,iter_max,conv_failure_switch,&settings);
    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,err_2,err_tol,iter_2,iter_max,conv_failure_switch,&settings);

    conv_failure_switch = 2;

    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,err_1,err_tol,iter_1,iter_max,conv_failure_switch,&settings);
    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,err_2,err_tol,iter_2,iter_max,conv_failure_switch,&settings);

    conv_failure_switch = 3; // error
    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,err_1,err_tol,iter_1,iter_max,conv_failure_switch,&settings);

    //
    // error_reporting_2

    conv_failure_switch = 0;

    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,conv_failure_switch,&settings);

    conv_failure_switch = 2;
    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,conv_failure_switch,&settings);

    conv_failure_switch = 3; // error
    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,conv_failure_switch,&settings);

    //
    // error_reporting_3

    conv_failure_switch = 0;

    optim::error_reporting(out_vals,x_p,optim_simple_fn_2,nullptr,success,err_1,err_tol,iter_1,iter_max,conv_failure_switch,&settings);
    optim::error_reporting(out_vals,x_p,optim_simple_fn_2,nullptr,success,err_2,err_tol,iter_2,iter_max,conv_failure_switch,&settings);

    conv_failure_switch = 1;

    optim::error_reporting(out_vals,x_p,optim_simple_fn_2,nullptr,success,err_1,err_tol,iter_1,iter_max,conv_failure_switch,&settings);
    optim::error_reporting(out_vals,x_p,optim_simple_fn_2,nullptr,success,err_2,err_tol,iter_2,iter_max,conv_failure_switch,&settings);

    conv_failure_switch = 2;

    optim::error_reporting(out_vals,x_p,optim_simple_fn_2,nullptr,success,err_1,err_tol,iter_1,iter_max,conv_failure_switch,&settings);
    optim::error_reporting(out_vals,x_p,optim_simple_fn_2,nullptr,success,err_2,err_tol,iter_2,iter_max,conv_failure_switch,&settings);

    conv_failure_switch = 3; // error
    optim::error_reporting(out_vals,x_p,optim_simple_fn_2,nullptr,success,err_1,err_tol,iter_1,iter_max,conv_failure_switch,&settings);

    // done

    std::cout << "\n     ***** End ERROR_REPORTING tests. *****     \n" << std::endl;

    return 0;
}
