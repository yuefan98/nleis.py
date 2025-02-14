====================
Release Notes
====================

Version 0.2
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