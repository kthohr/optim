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
 * Particle Swarm Optimization (PSO)
 *
 * Keith O'Hara
 * 08/04/2016
 *
 * This version:
 * 08/05/2017
 */

#include "optim.hpp"

bool
optim::ipso_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, double* value_out, opt_settings* settings_inp)
{
    bool success = false;

    const double BIG_POS_VAL = OPTIM_BIG_POS_NUM;
    const int n_vals = init_out_vals.n_elem;

    //
    //

    opt_settings settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int conv_failure_switch = settings.conv_failure_switch;
    const double err_tol = settings.err_tol;

    const int n_pop = (settings.pso_n_pop > 0) ? settings.pso_n_pop : 100;
    const int n_gen = (settings.pso_n_gen > 0) ? settings.pso_n_gen : 1000;

    double par_w = 1.0;
    const double par_w_max = 0.9;
    const double par_w_min = 0.4;
    const double par_damp = 0.99;
    double par_c_1 = 1.494;
    double par_c_2 = 1.494;

    const double par_initial_c1 = 2.5;
    const double par_final_c1 = 0.5;
    const double par_initial_c2 = 0.5;
    const double par_final_c2 = 2.5;

    const arma::vec par_initial_lb = ((int) settings.pso_lb.n_elem == n_vals) ? settings.pso_lb : arma::zeros(n_vals,1) - 0.5;
    const arma::vec par_initial_ub = ((int) settings.pso_ub.n_elem == n_vals) ? settings.pso_ub : arma::zeros(n_vals,1) + 0.5;

    arma::vec objfn_vals(n_pop);
    arma::mat P(n_pop,n_vals);

#ifdef OPTIM_OMP
    #pragma omp parallel for
#endif
    for (int i=0; i < n_pop; i++) {
        P.row(i) = init_out_vals.t() + par_initial_lb.t() + (par_initial_ub.t() - par_initial_lb.t())%arma::randu(1,n_vals);

        double prop_objfn_val = opt_objfn(P.row(i).t(),nullptr,opt_data);

        if (std::isnan(prop_objfn_val)) {
            prop_objfn_val = BIG_POS_VAL;
        }
        
        objfn_vals(i) = prop_objfn_val;
    }

    arma::vec best_vals = objfn_vals;

    arma::mat best_vecs = P;

    arma::mat V = arma::zeros(n_pop,n_vals);

    double global_best_val = objfn_vals.min();
    arma::rowvec global_best_vec = P.row( objfn_vals.index_min() );

    //
    //

    int iter = 0;
    double err = 2.0*err_tol;

    while (err > err_tol && iter < n_gen) {
        iter++;

        // arma::uvec rank_vec = arma::sort_index(objfn_vals, "descend");
        // arma::uvec rank_vec = arma::sort_index(objfn_vals, "ascend");

        // par_w = par_w_min + (par_w_max - par_w_min) * (iter + 1) / n_gen;

        par_c_1 = par_initial_c1 - (par_initial_c1 - par_final_c1) * (iter + 1) / n_gen;
        par_c_2 = par_initial_c2 - (par_initial_c2 - par_final_c2) * (iter + 1) / n_gen;

#ifdef OPTIM_OMP
        #pragma omp parallel for 
#endif
        for (int i=0; i < n_pop; i++) {
            // double w_i = par_w_min + (par_w_max - par_w_min)*(rank_vec(i) + 1) / n_pop;

            V.row(i) = par_w*V.row(i) + par_c_1*arma::randu(1,n_vals)%(best_vecs.row(i) - P.row(i)) + par_c_2*arma::randu(1,n_vals)%(global_best_vec - P.row(i));

            P.row(i) += V.row(i);
            
            //

            objfn_vals(i) = opt_objfn(P.row(i).t(),nullptr,opt_data);
                
            if (objfn_vals(i) < best_vals(i)) {
                best_vals(i) = objfn_vals(i);
                best_vecs.row(i) = P.row(i);
            }
        }

        if (best_vals.min() < global_best_val) {
            global_best_val = best_vals.min();
            global_best_vec = best_vecs.row( best_vals.index_min() );
        }

        par_w *= par_damp;
        // par_w = std::min(0.4,par_w*par_damp);
    }
    //
    error_reporting(init_out_vals,global_best_vec.t(),opt_objfn,opt_data,success,value_out,err,err_tol,iter,n_gen,conv_failure_switch);
    //
    return success;
}

bool
optim::ipso(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data)
{
    return ipso_int(init_out_vals,opt_objfn,opt_data,nullptr,nullptr);
}

bool
optim::ipso(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, opt_settings& settings)
{
    return ipso_int(init_out_vals,opt_objfn,opt_data,nullptr,&settings);
}

bool
optim::ipso(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, double& value_out)
{
    return ipso_int(init_out_vals,opt_objfn,opt_data,&value_out,nullptr);
}

bool
optim::ipso(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data)> opt_objfn, void* opt_data, double& value_out, opt_settings& settings)
{
    return ipso_int(init_out_vals,opt_objfn,opt_data,&value_out,&settings);
}
