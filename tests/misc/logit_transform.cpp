//
// coverage tests for logistic transform functions
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
    
    double val_in = 0.5;
    double val_tr = 0.0;
    double lb_in  = 0.0;
    double ub_in  = 1.0;

    arma::vec vals(2);
    vals.fill(0.5);

    arma::vec lb = arma::zeros(2,1);
    arma::vec ub = arma::ones(2,1);

    arma::vec vals_tr = arma::zeros(2,1);

    // done

    optim::logit_trans(vals,lb,ub);
    optim::logit_trans(vals);
    optim::logit_trans(val_in,lb_in,ub_in);

    optim::logit_inv_trans(vals_tr,lb,ub);
    optim::logit_inv_trans(vals_tr);
    optim::logit_inv_trans(val_tr,lb_in,ub_in);

    arma::vec vals2(2,1);
    vals2(0) = 0.0;
    vals2(1) = 1.0;

    optim::logit_trans(vals2,lb,ub); // this will trigger pars_trans.has_inf()

    return 0;
}
