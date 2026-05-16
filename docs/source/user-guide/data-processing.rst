Data Processing
===============

Use :mod:`nleis.data_processing` module to prepare frequency-domain EIS and 2nd-NLEIS
data before model fitting. The simplest workflow uses
:func:`data_truncation` to remove high-frequency
inductance effects and select the frequency range used for the 2nd-NLEIS
response.

The module also includes data-loading utilities for instruments that export
frequency-domain harmonic data or time-domain signals suitable for FFT-based
processing.

See :doc:`../api/data-processing` for the API reference.
