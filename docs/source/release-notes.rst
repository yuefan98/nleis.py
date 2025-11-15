====================
Release Notes
====================

Version 0.3 (2025-Nov-15)
---------------------------
This release brings some exciting new features and some corrections to previous versions, enabling a complete and efficient workflow 
for 2nd-NLEIS from data processing and validation to parallelization capabilities in analysis.

**What's Changed** `#31 <https://github.com/yuefan98/nleis.py/pull/31>`_

**Data Processing Module**
- Added a new data processing module (:code:`nleis.data_processing`) to facilitate the extraction of raw impedance data from time-domain, including filtering and normalization steps.

**Data Validation Module**
- Introduced a new data validation module (:code:`nleis.validation`) to ensure data integrity before analysis through the measurement model approach.
For more details, refer to our recent publication `Measurement Model Validation of Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy <https://iopscience.iop.org/article/10.1149/1945-7111/ae1064/meta>`_ in JECS

**Parallelization Support**
- Added parallel computing capabilities in fitting and analysis via the new module (:code:`nleis.parallel`), providing an efficient way to explore minimization landscapes with different initial guesses.
- New functions include :code:`multistart_fit`, :code:`batch_data_fit`, and :code:`batch_model_fit` to facilitate parallel fitting of multiple initial guesses, datasets, or models.

**Documentation Updates**
- Updated documentation with examples for data validation and parallel fitting.

**Fundamental governing equations correction**
- We introduced a 0.5 factor correction in all of our nonlinear circuit elements to align with the usage of sine or cosine perturbation in practical impedance experiments

**Code Improvements and Handling Improvements**
- We changed the default value of max_f from 10 Hz to np.inf to make it general for any impedance measurement.
- Changed data_processing function name to data_truncation to better reflect its functionality and moved it to the `data_processing` module.
- Restructured the automated tests to cover new modules and functionalities.
- Improved code structure and readability across modules.

**Full Changelog**: https://github.com/yuefan98/nleis.py/compare/v0.2...v0.3

Version 0.2 (2025-02-20)
---------------------------
This release introduces directed acyclic graph (DAG)-based function evaluation, optimizing computation speed and structure.
It also includes vectorized circuit calculations and new nonlinear circuit elements (:code:`RCDQ`, :code:`RCDQn`, :code:`RCSQ`, :code:`RCSQn`) for enhanced modeling flexibility. 🚀 

**What's Changed** `#27 <https://github.com/yuefan98/nleis.py/pull/27>`_

**Execution Graph**  

- Added support to graph-based function evaluation (:code:`graph = True`)
- `Documented <https://nleispy.readthedocs.io/en/latest/examples/graph_example.html>`_ speed benchmarks (at least 3x faster than :code:`eval()`)
- Updated compute method to return impedance values

**Computation Optimization**

- Vectorized circuit element calculations for faster performance

**New Circuit Elements**

- Added support to nonlinear RC with constant phase element: (:code:`RCDQ`, :code:`RCDQn`), and (:code:`RCSQ`, :code:`RCSQn`)

**New Contributors**

- Special thanks to @andersonjacob for the `source code <https://github.com/ECSHackWeek/impedance.py/pull/308>`_ 🎉 

**Full Changelog**: https://github.com/yuefan98/nleis.py/compare/v0.1.1...v0.2


Version 0.1.1 (2025-01-06)
---------------------------
This is the official release for the JOSS paper.

**What's Changed**

- Documentation updates by @dt-schwartz and @yuefan98 
- Bug fixes by @yuefan98 in `#25 <https://github.com/yuefan98/nleis.py/pull/25>`_

**Full Changelog**: https://github.com/yuefan98/nleis.py/compare/v0.1...v0.1.1

Version 0.1 (2024-09-26)
-------------------------
We are excited to announce the first official release of nleis.py! This release marks a significant step forward for nonlinear impedance analysis and will be submitted to JOSS for peer review.
 
**Key Features:**

- Simultaneous fitting and analysis of EIS and 2nd-NLEIS.
- Full support for nonlinear equivalent circuit (nECM) modeling and analysis.
- Various linear and nonlinear circuit element pairs derived from existing literature.
- Seamless integration with impedance.py for expanded impedance analysis capabilities.

**Improvements:**

- Comprehensive `documentation <https://nleispy.readthedocs.io/en/latest/>`_, including a Getting Started guide and API reference.
- Improved documentation for supported circuit elements.
- Improved code handling for better performance and readability. 

**Bug Fixes**

- Initial testing and issue resolution to ensure smooth functionality.

**New Contributors**

- Special thanks to @mdmurbach for joining the team and enhancing the package quality.

**Full Changelog**: https://github.com/yuefan98/nleis.py/commits/v0.1