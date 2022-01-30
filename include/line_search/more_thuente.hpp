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
 * Based on MINPACK fortran code and Dianne P. O'Leary's Matlab translation of MINPACK
 */

#ifndef _optim_more_thuente_HPP
#define _optim_more_thuente_HPP

namespace internal
{

fp_t
line_search_mt(
    fp_t step, 
    ColVec_t& x, 
    ColVec_t& grad, 
    const ColVec_t& direc, 
    const fp_t* wolfe_cons_1_inp, 
    const fp_t* wolfe_cons_2_inp, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data
);

// update the 'interval of uncertainty'

uint_t 
mt_step(
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
);

fp_t 
mt_sup_norm(
    const fp_t a, 
    const fp_t b, 
    const fp_t c
);

}

#endif
