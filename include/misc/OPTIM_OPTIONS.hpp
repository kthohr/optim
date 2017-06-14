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

#ifndef OPTIM_DEFAULT_PENALTY_GROWTH
    #define OPTIM_DEFAULT_PENALTY_GROWTH 2.0;
#endif

#ifndef OPTIM_CONV_FAILURE_POLICY
    #define OPTIM_CONV_FAILURE_POLICY 0;
#endif

// CG

#ifndef OPTIM_DEFAULT_CG_METHOD
    #define OPTIM_DEFAULT_CG_METHOD 2;
#endif

// DE

#ifndef OPTIM_DEFAULT_DE_NGEN
    #define OPTIM_DEFAULT_DE_NGEN 2000;
#endif

#ifndef OPTIM_DEFAULT_DE_F
    #define OPTIM_DEFAULT_DE_F 0.8;
#endif

#ifndef OPTIM_DEFAULT_DE_CR
    #define OPTIM_DEFAULT_DE_CR 0.9;
#endif

// Nelder-Mead

#ifndef OPTIM_DEFAULT_NM_ALPHA
    #define OPTIM_DEFAULT_NM_ALPHA 1.0;
#endif

#ifndef OPTIM_DEFAULT_NM_BETA
    #define OPTIM_DEFAULT_NM_BETA 2.0;
#endif

#ifndef OPTIM_DEFAULT_NM_GAMMA
    #define OPTIM_DEFAULT_NM_GAMMA 0.5;
#endif

#ifndef OPTIM_DEFAULT_NM_DELTA
    #define OPTIM_DEFAULT_NM_DELTA 0.5;
#endif
