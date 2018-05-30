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

/*
 * transform values
 */

inline
arma::vec
transform(const arma::vec& vals_inp, const arma::uvec& bounds_type, const arma::vec& lower_bounds, const arma::vec& upper_bounds)
{
    const size_t n_vals = bounds_type.n_elem;

    arma::vec vals_trans_out(n_vals);

    for (size_t i=0; i < n_vals; i++)
    {
        switch (bounds_type(i))
        {
            case 1: // no bounds
                vals_trans_out(i) = vals_inp(i);
                break;
            case 2: // lower bound only
                vals_trans_out(i) = std::log(vals_inp(i) - lower_bounds(i) + eps_dbl);
                break;
            case 3: // upper bound only
                vals_trans_out(i) = - std::log(upper_bounds(i) - vals_inp(i) + eps_dbl);
                break;
            case 4: // upper and lower bounds
                vals_trans_out(i) = std::log(vals_inp(i) - lower_bounds(i) + eps_dbl) - std::log(upper_bounds(i) - vals_inp(i) + eps_dbl);
                break;
        }
    }

    //

    return vals_trans_out;
}

inline
arma::vec
inv_transform(const arma::vec& vals_trans_inp, const arma::uvec& bounds_type, const arma::vec& lower_bounds, const arma::vec& upper_bounds)
{
    const size_t n_vals = bounds_type.n_elem;

    arma::vec vals_out(n_vals);

    for (size_t i=0; i < n_vals; i++)
    {
        switch (bounds_type(i))
        {
            case 1: // no bounds
                vals_out(i) = vals_trans_inp(i);
                break;
            case 2: // lower bound only
                if (!std::isfinite(vals_trans_inp(i)))
                {
                    vals_out(i) = lower_bounds(i) + eps_dbl;
                }
                else
                {
                    vals_out(i) = lower_bounds(i) + eps_dbl + std::exp(vals_trans_inp(i));
                }
                break;
            case 3: // upper bound only
                if (!std::isfinite(vals_trans_inp(i)))
                {
                    vals_out(i) = upper_bounds(i) - eps_dbl;
                }
                else
                {
                    vals_out(i) = upper_bounds(i) - eps_dbl - std::exp(-vals_trans_inp(i));
                }
                break;
            case 4: // upper and lower bounds
                if (!std::isfinite(vals_trans_inp(i)))
                {
                    if (std::isnan(vals_trans_inp(i)))
                    {
                        vals_out(i) = (upper_bounds(i) - lower_bounds(i)) / 2.0;
                    }
                    else if (vals_trans_inp(i) < 0.0)
                    {
                        vals_out(i) = lower_bounds(i) + eps_dbl;
                    }
                    else
                    {
                        vals_out(i) = upper_bounds(i) - eps_dbl;
                    }
                }
                else
                {
                    vals_out(i) = ( lower_bounds(i) + eps_dbl + (upper_bounds(i) - eps_dbl)*std::exp(vals_trans_inp(i)) ) \
                                    / ( 1.0 + std::exp(vals_trans_inp(i)) );

                    if (!std::isfinite(vals_out(i)))
                    {
                        vals_out(i) = upper_bounds(i) - eps_dbl;
                    }
                }
                break;
        }
    }

    //

    return vals_out;
}
