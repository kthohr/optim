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

#ifndef OPTIMLIB_STATS_SEED_VALUES
#define OPTIMLIB_STATS_SEED_VALUES

inline
size_t
generate_seed_value(const int ind_inp, const int n_threads, rand_engine_t& rand_engine)
{
    return static_cast<size_t>( (bmo_stats::runif<fp_t>(rand_engine) + ind_inp + n_threads) * 1000 );
    // return static_cast<size_t>( (ind_inp + n_threads) * 1000 );
}

#endif
