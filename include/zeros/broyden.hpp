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
 * Broyden's method for solving systems of nonlinear equations
 */

#ifndef _optim_broyden_HPP
#define _optim_broyden_HPP

// without jacobian

bool broyden_int(Vec_t& init_out_vals, 
                 std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                 void* opt_data, 
                 algo_settings_t* settings_inp);

bool broyden(Vec_t& init_out_vals, 
             std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
             void* opt_data);

bool broyden(Vec_t& init_out_vals, 
             std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
             void* opt_data, 
             algo_settings_t& settings);

// with jacobian

bool broyden_int(Vec_t& init_out_vals, 
                 std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                 void* opt_data,
                 std::function<Mat_t (const Vec_t& vals_inp, void* jacob_data)> jacob_objfn, 
                 void* jacob_data, 
                 algo_settings_t* settings_inp);

bool broyden(Vec_t& init_out_vals, 
             std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
             void* opt_data,
             std::function<Mat_t (const Vec_t& vals_inp, void* jacob_data)> jacob_objfn, 
             void* jacob_data);

bool broyden(Vec_t& init_out_vals, 
             std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
             void* opt_data,
             std::function<Mat_t (const Vec_t& vals_inp, void* jacob_data)> jacob_objfn, 
             void* jacob_data, 
             algo_settings_t& settings);

// derivative-free method

bool broyden_df_int(Vec_t& init_out_vals, 
                    std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                    void* opt_data, 
                    algo_settings_t* settings_inp);

bool broyden_df(Vec_t& init_out_vals, 
                std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                void* opt_data);

bool broyden_df(Vec_t& init_out_vals, 
                std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                void* opt_data, 
                algo_settings_t& settings);

// derivative-free method with jacobian

bool broyden_df_int(Vec_t& init_out_vals, 
                    std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, void* opt_data,
                    std::function<Mat_t (const Vec_t& vals_inp, void* jacob_data)> jacob_objfn, void* jacob_data, 
                    algo_settings_t* settings_inp);

bool broyden_df(Vec_t& init_out_vals, std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                void* opt_data,
                std::function<Mat_t (const Vec_t& vals_inp, void* jacob_data)> jacob_objfn, 
                void* jacob_data);

bool broyden_df(Vec_t& init_out_vals, std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                void* opt_data,
                std::function<Mat_t (const Vec_t& vals_inp, void* jacob_data)> jacob_objfn, 
                void* jacob_data, 
                algo_settings_t& settings);

// internal functions

double df_eta(uint_t k);
double df_proc_1(const Vec_t& x_vals, 
                 const Vec_t& direc, 
                 double sigma_1, 
                 uint_t k, 
                 std::function<Vec_t (const Vec_t& vals_inp, void* opt_data)> opt_objfn, 
                 void* opt_data);

#endif
