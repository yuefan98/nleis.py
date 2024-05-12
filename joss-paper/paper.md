---
title: 'nleis.py: A Nonlinear Electrochemical Impedance Analysis Toolbox'
tags:
  - Python
  - Electrochemistry
  - EIS
  - NLEIS
  - 2nd-NLEIS
  - Lithium-ion Batteries
  - Fuel Cells
authors:
  - name: Yuefan Ji
    email: yuefan@uw.edu
    orcid: 0000-0003-1912-767X
    affiliation: 1
  - name: Matthew D. Murbach
    orcid: 0000-0002-6583-5995
  - name: Daniel T. Schwartz
    corresponding: true 
    affiliation: 0000-0003-1173-5611

affiliations:
 - name: Department of Chemical Engineering & Clean Energy Institute, University of Washington, Seattle, WA, USA
   index: 1

date: 12 May 2024
bibliography: paper.bib
---

Building on the growing adoption of impedance.py `[@Murbach2020]` as an open-source software tool within the electrochemical impedance spectroscopy (EIS) community, nleis.py is a toolbox for impedance.py that aims to provide an easily accessible tool to perform second harmonic nonlinear EIS (2nd-NLEIS analysis), with the ability to extend to higher harmonic analysis in the future. The toolbox is designed with impedance.py in mind to minimize the learning curve for users. It inherits the basic functionality of impedance.py, introduces paired linear and 2nd-harmonic nonlinear circuit elements, and enables the simultaneous analysis of EIS and 2nd-NLEIS. With this toolbox, one can choose to individually analyze an EIS or 2nd-NLEIS spectra or perform simultaneous parameter estimation of linear and nonlinear impedance data using an impedance.py workflow. Ultimately, the nleis.py toolbox will be integrated into impedance.py as adoption grows, while maintaining the standalone version of nleis.py as a platform to develop advanced features as the field matures.

# Background

Electrochemical impedance spectroscopy (EIS) is a widely accepted electroanalytical method that is often used to characterize engineered electrochemical systems like fuel cells `[@Yuan2007]` and lithium-ion batteries (LIBs) `[@Meddings2020]`. EIS experiments and modeling require linearization of system response, leading to unavoidable information loss and model degeneracy challenges in real-world nonlinear electrochemical processes `[@orazem_eis_2008; @Fletcher1994]`. Second-harmonic nonlinear electrochemical impedance spectroscopy (2nd-NLEIS) is emerging as a powerful and complementary tool to EIS in lithium-ion battery (LIB) research. 2nd-NLEIS uses a moderately larger input modulation than conventional EIS to drive the electrochemical system into the weakly nonlinear regime where the fundamental frequency continues to represent the linear system response, and a small additional 2nd-harmonic signal adds key new information about the nonlinear dynamics of the interfaces under study `[@murbach_nleis_2018]`. Analyzing a 2nd-NLEIS signal unavoidably complicates the mathematical modeling compared to linear system theory, but it also provides a sensible way to break EIS degeneracy and generate key new insights into charge transfer, transport, and thermodynamic parameters that are inaccessible to EIS alone. Early work with the pseudo-two-dimensional (P2D) LIB model provided the first physical insights into the potential value of 2nd-NLEIS signals for battery research `[@murbach_p2d_2017]`, whereas quantitative parameter estimation of 2nd-NLEIS experiments has required adoption of physically-insightful reduced order models, such as Kirk et al.’s work developing a nonlinear single particle model (SPM) `[@kirk2023]`, and our work defining nonlinear Randles circuit (RC) and porous electrode models (PEM) `[@ji2023]`. 

# Statement of Need

As an emerging technique requiring nonlinear dynamic modeling to analyze experimental data, 2nd-NLEIS method adoption is slowed by a lack of commercial or open-source software for parameter estimation from experiments, even though NLEIS experiments can be performed with EIS equipment offered by several vendors `[@murbach_nleis_2018]`. Within the linear EIS community, the adoption of an open-source impedance.py equivalent circuit modeling workflow has successfully facilitated reproducible, easy-to-use, and transparent impedance analysis that supports an active community of users. By introducing nonlinear equivalent circuit modeling through nleis.py, we seek to enhance the accessibility of this powerful new technique with a streamlined data analysis pipeline that researchers are already familiar with, hence accelerating the co-development of theory and experiments. Moreover, there is neither a research nor industry standard platform available to measure and analyze 2nd-NLEIS. Consequently, we aim to use nleis.py as a starting point to establish 2nd-NLEIS measurements and analysis best-practices while working in concert with the impedance.py user community. 

# Naming Conventions and Parameter Assignments 

The linear and second harmonic nonlinear circuit elements are defined in a pair with an addition of ‘n’ after the nonlinear circuit element to facilitate the simultaneous analysis of linear and nonlinear impedance response. Additionally, because of the nature of the nonlinear response, the simplest possible circuit element is a wrapper Randles circuit (RC) rather than the Resistor (R) and capacitor (C) element defined in impedance.py. For example, the linear and nonlinear Randles circuits are defined in pairs as RC and RCn respectively. For higher harmonics (>2) analysis not yet implemented here, there is no restriction for the circuit element definition, but a general consideration is to avoid numbers in the definition of raw circuit elements for future expansion. Lastly, parameter assignment should follow a convention that first defines the linear parameters of a model (p_1) then the nonlinear parameters (p_2). For the RC circuit as an example, RC should only take [p_1] as parameter inputs, while RCn should take [p_1, p_2] as parameter inputs, as described in `[@ji2023]`.

# Current nleis.py Functionalities

## Nonlinear Equivalent Circuit Fitting

The 2nd-harmonic nonlinear Equivalent Circuit Fitting is accomplished with `NLEISCustomCircuit`. It inherits most features from the impedance.py `CustomCircuit`, but provides an extra level of flexibility for performing NLEIS specific tasks. Overall, the users should expect the same workflow as impedance.py.

## Simultaneous Equivalent Circuit Fitting of EIS and 2nd-NLEIS

`EISandNLEIS` is the key feature of nleis.py that enables the simultaneous analysis of EIS and 2nd-NLEIS with equivalent circuit modeling. The visual representation of nonlinear equivalent circuit representation can be found in `[@ji2023]`. Everything works like impedance.py, but the users should provide the correct pair of linear and nonlinear circuit strings with a single initial guess that is consistent with both linear and nonlinear circuit features seen in the data. For EIS and 2nd-NLEIS data with known error structure or relative magnitudes, the users can also specify the optimization weighting and normalization method for the EIS and 2nd-NLEIS data parameter estimation process, as introduced in `[@kirk2023; @ji2024]`.

## Visualization

The user can choose to use the plotting function in impedance.py or a customized plotting function for EIS (plot_first) and 2nd-NLEIS (plot_second) to get a correctly labeled Nyquist plot. 

## Nonlinear Circuit Elements

nleis.py supports a variety of linear and 2nd harmonic nonlinear circuit element pairs from simple Randles circuits to analytical porous electrode and numerical transmission line models. These models all rely on the foundation of the analytical theory developed by `[@ji2023]` for Randles and porous electrodes. Just like impedance.py, nleis.py supports manual element definition. If you want your model to be included in future releases, [create an issue](https://github.com/yuefan98/nleis.py/issues) on GitHub with your models to contribute to the project. 

# Future nleis.py Functionalities

## Data Processing

There is only a simple data processing function for frequency domain data truncation now. We hope to include data processing and conversion from either time or frequency domain data in the future (i.e. FFT capability for time domain data). If you have equipment that allows you to perform 2nd-NLEIS and wish it to be compatible with nleis.py, [create an issue](https://github.com/yuefan98/nleis.py/issues) with a sample file to help the advancement of 2nd-NLEIS.

## Data Validation

Unlike EIS, which already has a set of well-established validation methods available, there is not yet a standard set of tools to quickly validate the causality and stationarity requirements of 2nd-NLEIS data. We are actively working on the development of such a data validation method, which will be incorporated in the future. If you have a new method for 2nd-NLEIS validation, we are also interested in including it in the future after it is peer reviewed and published. [Create an issue](https://github.com/yuefan98/nleis.py/issues) or [submit a pull request](https://github.com/yuefan98/nleis.py/pulls) to initiate review for inclusion.

## Contribute to the Project

In general, 2nd-NLEIS is a novel technique for electrochemical science and engineering research and development. Many areas familiar to EIS analysts are not fully developed for nonlinear systems. If you are publishing theoretical or experimental work that is advancing the field and would like to disseminate it as software others can use as part of nleis.py, we encourage you to [create an issue](https://github.com/yuefan98/nleis.py/issues) on GitHub and become a contributor to nleis.py. 

# Side-by-Side Comparison between impedance.py and nleis.py API

nleis.py comes with detailed documentation with examples and concepts for new users to [get started](https://nleispy.readthedocs.io/en/latest/getting-started.html). For existing impedance.py users, we expect this side-by-side comparison between impedance.py and nleis.py can reduce barriers for extending your impedance analysis to the weakly nonlinear regime.

![Side-by-side comparison between `impedance.py` and `nleis.py` API](API_comparison.png) 

# Acknowledgments

We thank An-Hung Shih and Lauren Frank for their preliminary testing and invaluable feedback. An up-to-date list of contributors can be found on GitHub.

# References