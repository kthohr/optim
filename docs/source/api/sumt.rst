.. Copyright (c) 2016-2022 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Sequential Unconstrained Minimization Technique
===============================================

**Table of contents**

.. contents:: :local:

----

Description
-----------

For a general problem

.. math::

   \min_x f(x) \text{ subject to } g_k (x) \leq 0, \ \ k \in \{1, \ldots, K \}

The Sequential Unconstrained Minimization Technique solves:

.. math::

   \min_x \left\{ f(x) + c(i) \times \frac{1}{2} \sum_{k=1}^K \left( \max \{ 0, g_k(x) \} \right)^2 \right\}

The algorithm stops when the error is less than ``err_tol``, or the total number of 'generations' exceeds a desired (or default) value.

----

Definitions
-----------

.. _sumt-func-ref1:
.. doxygenfunction:: sumt(ColVec_t& init_out_vals, std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, void* opt_data, std::function<ColVec_t (const ColVec_t& vals_inp, Mat_t* jacob_out, void* constr_data)> constr_fn, void* constr_data)
   :project: optimlib

.. _sumt-func-ref2:
.. doxygenfunction:: sumt(ColVec_t& init_out_vals, std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, void* opt_data, std::function<ColVec_t (const ColVec_t& vals_inp, Mat_t* jacob_out, void* constr_data)> constr_fn, void* constr_data, algo_settings_t& settings)
   :project: optimlib

----

Examples
--------


