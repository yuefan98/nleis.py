# import warnings
# from scipy.linalg import inv
# from scipy.optimize import basinhopping

import numpy as np
from scipy.optimize import curve_fit
from impedance.models.circuits.elements import circuit_elements, \
    get_element_from_name
from impedance.models.circuits.fitting import check_and_eval
from .fitting import set_default_bounds, buildCircuit, extract_circuit_elements
from scipy.optimize import minimize
import warnings
from nleis.fitting import CircuitGraph

# Customize warning format (here, simpler and just the message)
warnings.formatwarning = lambda message, category, filename, lineno, \
    line=None: f'{category.__name__}: {message}\n'

ints = '0123456789'


def simul_fit(frequencies, Z1, Z2, circuit_1, circuit_2, edited_circuit,
              initial_guess, constants_1={}, constants_2={},
              bounds=None, opt='max', cost=0.5, max_f=10, param_norm=True,
              positive=True, graph=False, **kwargs):
    """
    Main function for the simultaneous fitting of EIS and 2nd-NLEIS data.

    This function fits the equivalent circuits for both EIS and 2nd-NLEIS data
    simultaneously using `scipy.optimize.curve_fit
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_
    by default. It supports
    parameter normalization, bounds customization, and two optimization methods
    (`'max'` for maximum normalization and `'neg'` for negative log-likelihood)

    Parameters
    ----------
    frequencies : numpy.ndarray
        Array of frequency values.

    Z1 : numpy.ndarray of dtype complex128
        EIS data (complex impedance values).

    Z2 : numpy.ndarray of dtype complex128
        2nd-NLEIS data (complex impedance values).

    circuit_1 : str
        String defining the EIS equivalent circuit to be fit.

    circuit_2 : str
        String defining the 2nd-NLEIS equivalent circuit to be fit.

    edited_circuit : str
        Edited circuit string that is applied to both EIS and NLEIS fitting.

        Example:

        circuit_1 = L0-R0-TDP0-TDS1

        circuit_2 = TDPn0-TDSn1

        edited_circuit = L0-R0-TDPn0-TDSn1

    initial_guess : list of float
        Initial guesses for the fit parameters.

    constants_1 : dict, optional
        Parameters and their values to hold constant during EIS fitting.
        Defaults to an empty dictionary.

    constants_2 : dict, optional
        Parameters and their values to hold constant during 2nd-NLEIS fitting.
        Defaults to an empty dictionary.

    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. If not provided, default bounds
        will be set based on the circuit elements.

        Special bounds:

        - CPE alpha has an upper bound of 1.

        - Symmetry parameter (ε) for 2nd-NLEIS has bounds between -0.5 and 0.5.

        - Curvature parameter (κ) for 2nd-NLEIS has bounds
          between -np.inf and np.inf.

    opt : str, optional
        Optimization method. Default is `'max'` for maximum normalization.
        Negative log-likelihood is also supported as `'neg'`.

    cost : float, optional
        Weighting between EIS and 2nd-NLEIS data. A value greater than 0.5
        puts more weight on EIS,
        while a value less than 0.5 puts more weight on 2nd-NLEIS.
        Default is 0.5.

    max_f : int, optional
        Maximum frequency of interest for 2nd-NLEIS. Default is 10.

    param_norm : bool, optional
        If True, parameter normalization is applied to improve convergence.
        Defaults to True.

    positive : bool, optional
        If True, high-frequency inductance is eliminated. Defaults to True.

    graph : bool, optional
        Whether to use execution graph to process the circuit.
        Defaults to False, which uses eval based code

    kwargs :
        Additional keyword arguments passed to `scipy.optimize.curve_fit`.

    Returns
    -------
    p_values : list of float
        Best-fit parameters for EIS and 2nd-NLEIS data.

    p_errors : list of float
        One standard deviation error estimates for fitting parameters.
        If using `'neg'` optimization, this value will be None.

    """
    # Todo improve the the negtive loglikelihood,
    # the code works fine for RC but not porous electrode

    # set upper and lower bounds on a per-element basis
    if bounds is None:
        combined_constant = constants_2.copy()
        combined_constant.update(constants_1)
        bounds = set_default_bounds(
            edited_circuit, constants=combined_constant)
        ub = np.ones(len(bounds[1]))
    else:
        if param_norm:
            inf_in_bounds = np.any(np.isinf(bounds[0])) \
                or np.any(np.isinf(bounds[1]))
            if inf_in_bounds:
                lb = np.where(np.array(bounds[0]) == -np.inf, -1e10, bounds[0])
                ub = np.where(np.array(bounds[1]) == np.inf, 1e10, bounds[1])
                bounds = (lb, ub)
                warnings.warn("inf is detected in the bounds, "
                              "to enable parameter normalization, "
                              "the bounds has been capped at 1e10. "
                              "You can disable parameter normalization "
                              "by set param_norm to False .")
            else:
                ub = bounds[1]

            bounds = bounds/ub
        else:
            ub = np.ones(len(bounds[1]))

    initial_guess = initial_guess/ub

    if positive:
        mask1 = np.array(Z1.imag) < 0
        frequencies = frequencies[mask1]
        Z1 = Z1[mask1]
        Z2 = Z2[mask1]
        mask2 = np.array(frequencies) < max_f
        Z2 = Z2[mask2]
    else:
        mask2 = np.array(frequencies) < max_f
        Z2 = Z2[mask2]

    Z1stack = np.hstack([Z1.real, Z1.imag])
    Z2stack = np.hstack([Z2.real, Z2.imag])
    Zstack = np.hstack([Z1stack, Z2stack])
    # weighting scheme for fitting
    if opt == 'max':
        if 'maxfev' not in kwargs:
            kwargs['maxfev'] = 1e5
        if 'ftol' not in kwargs:
            kwargs['ftol'] = 1e-13
        Z1max = max(np.abs(Z1))
        Z2max = max(np.abs(Z2))

        sigma1 = np.ones(len(Z1stack))*Z1max/(cost**0.5)
        sigma2 = np.ones(len(Z2stack))*Z2max/((1-cost)**0.5)
        kwargs['sigma'] = np.hstack([sigma1, sigma2])

        popt, pcov = curve_fit(
            wrapCircuit_simul(edited_circuit, circuit_1, constants_1,
                              circuit_2, constants_2,
                              ub, max_f, graph=graph), frequencies, Zstack,
            p0=initial_guess, bounds=bounds, **kwargs)

    # Calculate one standard deviation error estimates for fit parameters,
    # defined as the square root of the diagonal of the covariance matrix.
    # https://stackoverflow.com/a/52275674/5144795
    # and the following for the bounded and normalized case
    # https://stackoverflow.com/questions/14854339/in-scipy-how-and-why-does-curve-fit-calculate-the-covariance-of-the-parameter-es
        perror = np.sqrt(np.diag(ub*pcov*ub.T))

        return popt*ub, perror
    if opt == 'neg':
        # This method does not provides converge solution
        # for porous electrode model under current development
        # The method is introduced by Kirk et al, J. Electrochem. Soc. (2023)
        # doi: 10.1149/1945-7111/acada7
        # to support analysis for single particle model

        bounds = tuple(tuple((bounds[0][i], bounds[1][i]))
                       for i in range(len(bounds[0])))

        res = minimize(
            wrapNeg_log_likelihood(frequencies, Z1, Z2, edited_circuit,
                                   circuit_1, constants_1,
                                   circuit_2, constants_2,
                                   ub, max_f, cost=cost, graph=graph),
            x0=initial_guess, bounds=bounds, **kwargs)

        return (res.x*ub, None)


def wrapNeg_log_likelihood(frequencies, Z1, Z2, edited_circuit,
                           circuit_1, constants_1,
                           circuit_2, constants_2, ub, max_f=10, cost=0.5,
                           graph=False):
    ''' wraps function so we can pass the circuit string
    for negtive log likelihood optimization'''

    def wrappedNeg_log_likelihood(parameters):
        """ returns a stacked array of real and imaginary impedance
        components

        Parameters
        ----------
        frequencies: list of floats
        Z1: EIS data
        Z2: NLEIS data
        edited_circuit : string
        circuit_1 : string
        constants_1 : dict
        circuit_2 : string
        constants_2 : dict
        ub : list of floats upper bound if bounds are provided
        max_f: int
        cost : float, optional
            Weighting between EIS and 2nd-NLEIS data. A value greater than 0.5
            puts more weight on EIS,
            while a value less than 0.5 puts more weight on 2nd-NLEIS.
            Default is 0.5.
        graph : bool, optional
            Whether to use execution graph to process the circuit.
            Defaults to False, which uses eval based code
        parameters : list of floats

        Returns
        -------
        array of floats

        """
        f1 = frequencies
        mask = np.array(frequencies) < max_f
        f2 = frequencies[mask]
        x1, x2 = wrappedImpedance(edited_circuit,
                                  circuit_1, constants_1, circuit_2,
                                  constants_2, f1, f2, parameters*ub,
                                  graph=graph)

        # No normalization in currently applied
        # Z1max = max(np.abs(Z1))
        # Z2max = max(np.abs(Z2))
        # log1 = np.log(sum(((Z1.real-x1.real)/Z1max)**2))
        # +np.log(sum(((Z1.imag-x1.imag)/Z1max)**2))
        # log2 = np.log(sum(((Z2.real-x2.real)/Z2max)**2))
        # +np.log(sum(((Z2.imag-x2.imag)/Z2max)**2))

        log1 = np.log(sum(abs(Z1-x1)**2))
        log2 = np.log(sum(abs(Z2-x2)**2))
        return (cost*log1+(1-cost)*log2)
    return wrappedNeg_log_likelihood


def wrapCircuit_simul(edited_circuit, circuit_1, constants_1, circuit_2,
                      constants_2, ub, max_f=10, graph=False):
    """ wraps function so we can pass the circuit string
    for simultaneous fitting """
    def wrappedCircuit_simul(frequencies, *parameters):
        """ returns a stacked array of real and imaginary impedance
        components

        Parameters
        ----------
        edited_circuit : string
        circuit_1 : string
        constants_1 : dict
        circuit_2 : string
        constants_2 : dict
        ub : list of floats upper bound if bounds are provided
        max_f: float
        graph : bool, optional
            Whether to use execution graph to process the circuit.
            Defaults to False, which uses eval based code

        frequencies : list of floats

        parameters : list of floats


        Returns
        -------
        array of floats

        """

        f1 = frequencies
        mask = np.array(frequencies) < max_f
        f2 = frequencies[mask]
        x1, x2 = wrappedImpedance(edited_circuit,
                                  circuit_1, constants_1,
                                  circuit_2, constants_2,
                                  f1, f2, parameters*ub, graph=graph)

        y1_real = np.real(x1)
        y1_imag = np.imag(x1)
        y1_stack = np.hstack([y1_real, y1_imag])
        y2_real = np.real(x2)
        y2_imag = np.imag(x2)
        y2_stack = np.hstack([y2_real, y2_imag])

        return np.hstack([y1_stack, y2_stack])
    return wrappedCircuit_simul


def wrappedImpedance(edited_circuit, circuit_1, constants_1, circuit_2,
                     constants_2, f1, f2, parameters, graph=False):
    """
    Calculate EIS and 2nd-NLEIS impedances using the provided circuits.

    This function evaluates the equivalent circuit models for both EIS and
    2nd-NLEIS using the provided frequencies and parameters.

    Parameters
    ----------
    edited_circuit : str
        The edited circuit string which contains
        the combined elements of EIS and 2nd-NLEIS.

    circuit_1 : str
        The equivalent circuit string for EIS.

    constants_1 : dict
        Constants used in the EIS circuit model.

    circuit_2 : str
        The equivalent circuit string for 2nd-NLEIS.

    constants_2 : dict
        Constants used in the 2nd-NLEIS circuit model.

    f1 : list of float
        frequencies  for EIS.

    f2 : list of float
        frequencies for 2nd-NLEIS.

    parameters : list of float
        Full set of parameters derived from the edited circuit string.
    graph : bool, optional
        Whether to use execution graph to process the circuit.
        Defaults to False, which uses eval based code

    Returns
    -------
    x1 : numpy.ndarray
        Calculated impedance for EIS (Z1).

    x2 : numpy.ndarray
        Calculated impedance for 2nd-NLEIS (Z2).
    """

    p1, p2 = individual_parameters(
        edited_circuit, parameters, constants_1, constants_2)
    if graph:
        graph_EIS = CircuitGraph(circuit_1, constants_1)
        graph_NLEIS = CircuitGraph(circuit_2, constants_2)
        x1 = graph_EIS.compute(f1, *p1)
        x2 = graph_NLEIS.compute(f2, *p2)
    else:
        x1 = eval(buildCircuit(circuit_1, f1, *p1,
                               constants=constants_1, eval_string='',
                               index=0)[0],
                  circuit_elements)
        x2 = eval(buildCircuit(circuit_2, f2, *p2,
                               constants=constants_2, eval_string='',
                               index=0)[0],
                  circuit_elements)
    return (x1, x2)


def individual_parameters(edited_circuit,
                          parameters, constants_1, constants_2):
    """
    Separate parameters for EIS and 2nd-NLEIS based on the edited circuit.

    This function parses the edited circuit string and assigns the parameters
    to either EIS or 2nd-NLEIS depending on the element type.

    Parameters
    ----------
    edited_circuit : str
        Edited circuit string combining both EIS and 2nd-NLEIS elements.
        For example, if the EIS string is 'L0-R0-TDS0-TDS1' and the 2nd-NLEIS
        string is 'd(TDSn0-TDSn1)',
        the edited string becomes 'L0-R0-TDSn0-TDSn1'.

    parameters : list of float
        Full set of parameters corresponding to the edited circuit string.

    constants_1 : dict
        Constants for the EIS circuit elements.

    constants_2 : dict
        Constants for the 2nd-NLEIS circuit elements.

    Returns
    -------
    p1 : list of float
        Parameters for EIS.

    p2 : list of float
        Parameters for 2nd-NLEIS.
    """

    if edited_circuit == '':
        return [], []
    parameters = list(parameters)
    elements_1 = extract_circuit_elements(edited_circuit)
    p1 = []
    p2 = []
    index = 0
    # Parse elements and store values
    for elem in elements_1:

        raw_elem = get_element_from_name(elem)
        nleis_elem_number = check_and_eval(raw_elem).num_params
        # this might be improvable, but depends on
        # how we want to define the name
        if (elem[0] == 'T' or elem[0:2] == 'RC') and 'n' in elem:
            # check for nonlinear element
            eis_elem_number = check_and_eval(raw_elem[0:-1]).num_params

        else:
            eis_elem_number = nleis_elem_number

        for j in range(nleis_elem_number):
            if eis_elem_number > 1:
                if j < eis_elem_number:
                    eis_current_elem = elem.replace("n", "") + '_{}'.format(j)
                else:
                    eis_current_elem = None
                nleis_current_elem = elem + '_{}'.format(j)

            else:
                eis_current_elem = elem
                nleis_current_elem = elem
            if eis_current_elem in constants_1.keys():
                continue
            elif (nleis_current_elem in constants_2.keys() and
                  eis_current_elem not in constants_1.keys()):
                continue
            else:
                if eis_elem_number == 1:
                    p1.append(parameters[index])

                elif eis_elem_number > 1 and j < eis_elem_number:
                    p1.append(parameters[index])
                    if nleis_elem_number > eis_elem_number:
                        p2.append(parameters[index])
                else:
                    p2.append(parameters[index])

                index += 1

    return p1, p2
