Data Validation
===============

The :mod:`nleis.validation` module provides measurement-model tools for checking whether
EIS and 2nd-NLEIS spectra satisfy the assumptions needed for reliable
interpretation. These tools support both cost-based and confidence-interval
based model selection workflows.

Use validation before interpreting fitted model parameters, especially when
working with experimental 2nd-NLEIS data where the nonlinear signal is small
relative to the EIS response.

See :doc:`../examples/validation` for a complete validation workflow and
:doc:`../api/validation` for function-level details.
