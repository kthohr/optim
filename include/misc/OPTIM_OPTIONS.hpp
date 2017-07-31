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

#pragma once

// basic settings

#ifndef OPTIM_BIG_POS_NUM
    #define OPTIM_BIG_POS_NUM 1E09;
#endif

#ifndef OPTIM_DEFAULT_ERR_TOL
    #define OPTIM_DEFAULT_ERR_TOL 1E-08;
#endif

#ifndef OPTIM_DEFAULT_ITER_MAX
    #define OPTIM_DEFAULT_ITER_MAX 2000;
#endif

#ifndef OPTIM_CONV_FAILURE_POLICY
    #define OPTIM_CONV_FAILURE_POLICY 0;
#endif

// CG

#ifndef OPTIM_DEFAULT_CG_METHOD
    #define OPTIM_DEFAULT_CG_METHOD 2;
#endif

#ifndef OPTIM_DEFAULT_CG_RESTART_THRESHOLD
    #define OPTIM_DEFAULT_CG_RESTART_THRESHOLD 0.1;
#endif

// DE

#ifndef OPTIM_DEFAULT_DE_NGEN
    #define OPTIM_DEFAULT_DE_NGEN 2000;
#endif

#ifndef OPTIM_DEFAULT_DE_CHECK_FREQ
    #define OPTIM_DEFAULT_DE_CHECK_FREQ 20;
#endif

#ifndef OPTIM_DEFAULT_DE_PAR_F
    #define OPTIM_DEFAULT_DE_PAR_F 0.8;
#endif

#ifndef OPTIM_DEFAULT_DE_PAR_CR
    #define OPTIM_DEFAULT_DE_PAR_CR 0.9;
#endif

// Nelder-Mead

#ifndef OPTIM_DEFAULT_NM_PAR_ALPHA
    #define OPTIM_DEFAULT_NM_PAR_ALPHA 1.0; // reflection parameter
#endif

#ifndef OPTIM_DEFAULT_NM_PAR_BETA
    #define OPTIM_DEFAULT_NM_PAR_BETA 0.5; // contraction parameter
#endif

#ifndef OPTIM_DEFAULT_NM_PAR_GAMMA
    #define OPTIM_DEFAULT_NM_PAR_GAMMA 2.0; // expansion parameter
#endif

#ifndef OPTIM_DEFAULT_NM_PAR_DELTA
    #define OPTIM_DEFAULT_NM_PAR_DELTA 0.5; // shrinkage parameter
#endif

// SUMT

#ifndef OPTIM_DEFAULT_SUMT_PENALTY_GROWTH
    #define OPTIM_DEFAULT_SUMT_PENALTY_GROWTH 2.0;
#endif
