.. Copyright (c) 2016-2020 Keith O'Hara

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

   \min_x f(x) \text{ subject to } g_k (x) \geq 0, \ \ k \in \{1, \ldots, K \}

The Sequential Unconstrained Minimization Technique solves:

.. math::

   \min_x \left\{ f(x) + c(i) \times \frac{1}{2} \sum_{k=1}^K \left( \max \{ 0, g_k(x) \} \right)^2 \right\}

The algorithm stops when is less than err_tol, or the total number of 'generations' exceeds a desired (or default) value.

----

Definitions
-----------

.. _sumt-func-ref1:
.. doxygenfunction:: sumt(Vec_t&, std::function<doubleconst Vec_t &vals_inp, Vec_t *grad_out, void *opt_data>, void *, std::function<Vec_tconst Vec_t &vals_inp, Mat_t *jacob_out, void *constr_data>, void *)
   :project: optimlib

.. _sumt-func-ref2:
.. doxygenfunction:: sumt(Vec_t&, std::function<doubleconst Vec_t &vals_inp, Vec_t *grad_out, void *opt_data>, void *, std::function<Vec_tconst Vec_t &vals_inp, Mat_t *jacob_out, void *constr_data>, void *, algo_settings_t&)
   :project: optimlib

----

Examples
--------


