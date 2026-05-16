Fitting Workflows
=================

The package supports standalone 2nd-NLEIS fitting with
:class:`NLEISCustomCircuit` class and simultaneous EIS/2nd-NLEIS fitting
with :class:`EISandNLEIS` class.

For simultaneous fitting, provide paired linear and nonlinear circuit strings,
one initial parameter vector, and optional constants or bounds. The fitting
workflow then estimates shared and nonlinear parameters while preserving the
relationship between the EIS and 2nd-NLEIS responses.

See :doc:`../examples/nleis_example` for the standard fitting workflow,
:doc:`../examples/graph_example` for graph-based execution, and
:doc:`../api/nleis` for class details.
