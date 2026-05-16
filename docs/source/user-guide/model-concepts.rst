Model Concepts
==============

:code:`nleis.py` follows the equivalent circuit workflow familiar from
:code:`impedance.py`, but nonlinear circuit elements are defined as paired
linear and 2nd-NLEIS responses. The nonlinear element name is distinguished by
an added :code:`n`, such as :code:`RCn` for the nonlinear counterpart to
:code:`RC`.

For simultaneous analysis, :class:`EISandNLEIS` class accepts one circuit
string for EIS and one paired circuit string for 2nd-NLEIS. Shared parameters
are aligned internally so that the fitted result can be reported for each
response.

See :doc:`../examples/nleis_example` for a complete model-building workflow and
:doc:`../api/nleis-elements-pair` for the available circuit element pairs.
