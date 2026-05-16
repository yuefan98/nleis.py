Parallel Workflows
==================

The :mod:`nleis.parallel` module provides utilities for running repeated fitting tasks,
including multistart fitting, batch fitting across datasets, and batch fitting
across model candidates.

Use these workflows when the fit landscape is sensitive to initial guesses or
when multiple datasets or model structures need to be compared consistently.

See :doc:`../examples/multistart`, :doc:`../examples/batch_data`, and
:doc:`../examples/batch_model` for examples. See :doc:`../api/parallel` for
the API reference.
