//
// coverage tests for error_reporting
//
// g++-mp-7 -O2 -Wall -std=c++11 -I/opt/local/include error_reporting_test.cpp -o error_reporting.test -L/opt/local/lib -loptim -framework Accelerate
// g++-mp-7 -O2 -Wall -std=c++11 -I./../../include error_reporting.cpp -o error_reporting.test -L./../.. -loptim -framework Accelerate
//

#include "optim.hpp"

double optim_simple_fn_1(const arma::vec& vals_inp, arma::vec* grad, void* opt_data)
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
    double value_out = 1.0;

    double err_1 = 0.5;
    double err_2 = 1.5;
    double err_tol = 1.0;

    int iter_1 = 1;
    int iter_2 = 3;
    int iter_max = 2;

    //
    // error_reporting_1

    int conv_failure_switch = 0;

    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,&value_out,err_1,err_tol,iter_1,iter_max,conv_failure_switch);
    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,&value_out,err_2,err_tol,iter_2,iter_max,conv_failure_switch);

    conv_failure_switch = 1;

    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,&value_out,err_1,err_tol,iter_1,iter_max,conv_failure_switch);
    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,&value_out,err_2,err_tol,iter_2,iter_max,conv_failure_switch);

    conv_failure_switch = 2;

    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,&value_out,err_1,err_tol,iter_1,iter_max,conv_failure_switch);
    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,&value_out,err_2,err_tol,iter_2,iter_max,conv_failure_switch);

    conv_failure_switch = 3; // error
    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,&value_out,err_1,err_tol,iter_1,iter_max,conv_failure_switch);

    //
    // error_reporting_2

    conv_failure_switch = 0;

    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,&value_out,conv_failure_switch);

    conv_failure_switch = 2;
    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,&value_out,conv_failure_switch);

    conv_failure_switch = 3; // error
    optim::error_reporting(out_vals,x_p,optim_simple_fn_1,nullptr,success,&value_out,conv_failure_switch);

    //
    // error_reporting_3

    conv_failure_switch = 0;
    arma::vec val_vec_out;

    optim::error_reporting(out_vals,x_p,optim_simple_fn_2,nullptr,success,&val_vec_out,err_1,err_tol,iter_1,iter_max,conv_failure_switch);
    optim::error_reporting(out_vals,x_p,optim_simple_fn_2,nullptr,success,&val_vec_out,err_2,err_tol,iter_2,iter_max,conv_failure_switch);

    conv_failure_switch = 1;

    optim::error_reporting(out_vals,x_p,optim_simple_fn_2,nullptr,success,&val_vec_out,err_1,err_tol,iter_1,iter_max,conv_failure_switch);
    optim::error_reporting(out_vals,x_p,optim_simple_fn_2,nullptr,success,&val_vec_out,err_2,err_tol,iter_2,iter_max,conv_failure_switch);

    conv_failure_switch = 2;

    optim::error_reporting(out_vals,x_p,optim_simple_fn_2,nullptr,success,&val_vec_out,err_1,err_tol,iter_1,iter_max,conv_failure_switch);
    optim::error_reporting(out_vals,x_p,optim_simple_fn_2,nullptr,success,&val_vec_out,err_2,err_tol,iter_2,iter_max,conv_failure_switch);

    conv_failure_switch = 3; // error
    optim::error_reporting(out_vals,x_p,optim_simple_fn_2,nullptr,success,&val_vec_out,err_1,err_tol,iter_1,iter_max,conv_failure_switch);

    // done

    return 0;
}
