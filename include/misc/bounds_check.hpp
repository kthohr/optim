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
 * check and impose sampling bounds for DE and PSO
 */

inline
void
sampling_bounds_check(
    const bool vals_bound,
    const size_t n_vals,
    const ColVecInt_t bounds_type,
    const ColVec_t& hard_lower_bounds, 
    const ColVec_t& hard_upper_bounds,
    ColVec_t& sampling_lower_bounds,
    ColVec_t& sampling_upper_bounds
)
{
    if (vals_bound) {
        for (size_t i = 0; i < n_vals; ++i) {
            if (bounds_type(i) == 4 || bounds_type(i) == 2) {
                // lower and upper bound imposed || lower bound only
                sampling_lower_bounds(i) = std::max( hard_lower_bounds(i), sampling_lower_bounds(i) );
            }
            if (bounds_type(i) == 4 || bounds_type(i) == 3) {
                // lower and upper bound imposed || upper bound only
                sampling_upper_bounds(i) = std::min( hard_upper_bounds(i), sampling_upper_bounds(i) );
            }
        }
    }
}
