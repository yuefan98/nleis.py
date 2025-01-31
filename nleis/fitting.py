import warnings

import numpy as np
from scipy.linalg import inv
from scipy.optimize import curve_fit, basinhopping
from impedance.models.circuits.elements import circuit_elements
from impedance.models.circuits.elements import get_element_from_name
from impedance.models.circuits.fitting import check_and_eval, rmse

import networkx as nx
import re

# Note: a lot of codes are directly adopted from impedance.py.,
# which is designed to be enable a easy integration in the future,
# but now we are keep them separated to ensure the stable performance

ints = '0123456789'


def mae(a, b):
    """
    Calculate the Mean Absolute Error (MAE) between two arrays.

    Parameters
    ----------
    a : numpy.ndarray
        Array of experimental data.

    b : numpy.ndarray
        Array of model fit data.

    Returns
    -------
    float
        The calculated Mean Absolute Error between `a` and `b`.
    """
    return (np.mean(abs(a-b)))


def mape(a, b):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between two arrays.

    Parameters
    ----------
    a : numpy.ndarray
        Array of experimental data.

    b : numpy.ndarray
        Array of model fit data.

    Returns
    -------
    float
        The calculated Mean Absolute Percentage Error (MAPE)
        between `a` and `b`, expressed as a percentage.
    """

    return (np.mean(abs(a-b)/abs(a))*100)


def seq_fit_param(input_dic, target_arr, output_arr):
    '''
    Convert obtained EIS results to a constant dictionary for
    2nd-NLEIS analysis using sequential optimization as discussed in [1].

    Parameters
    ----------
    input_dic : dict
        Dictionary of EIS fitting results.

    target_arr : list of str
        A list of EIS circuit elements that need to be converted.
        Example: ['TDS0', 'TDS1'].

    output_arr : list of str
        A list of 2nd-NLEIS circuit elements that will be converted to.
        Example: ['TDSn0', 'TDSn1'].

    Raises
    ------
    ValueError
        Raised if the `target_arr` and `output_arr` have different lengths.

    Returns
    -------
    output_dic : dict
        A dictionary of constants that can be used for sequential 2nd-NLEIS
        optimization.

    References
    ----------
    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    II.Model-Based Analysis of Lithium-Ion Battery Experiments.
    J. Electrochem. Soc., 2024. `doi:10.1149/1945-7111/ad2596
    <https://iopscience.iop.org/article/10.1149/1945-7111/ad2596>`_.

    '''
    output_dic = {}
    n = len(target_arr)
    if n != len(output_arr):
        raise ValueError(
            'Target Array and Output Array Must Have the Same Length ')

    for i in range(0, n):
        for j in range(check_and_eval(target_arr[i][0:-1]).num_params):
            if target_arr[i]+'_'+str(j) not in input_dic:
                continue
            output_dic[output_arr[i]+'_' +
                       str(j)] = input_dic[target_arr[i]+'_'+str(j)]
    return (output_dic)


def set_default_bounds(circuit, constants={}):
    """
    Set default bounds for optimization.

    This function assigns default bounds for all parameters in the given
    circuit. The default lower bound is 0, and the upper bound is `np.inf`.

    Exceptions:

    - CPE and La alpha have an upper bound of 1.

    - Symmetry parameter (ε) for 2nd-NLEIS has bounds between -0.5 and 0.5.

    - Curvature parameter (κ) for 2nd-NLEIS has bounds
      between -np.inf and np.inf.

    Parameters
    ----------
    circuit : str
        String defining the equivalent circuit to be fit.

    constants : dict, optional
        Dictionary of parameters and their values to be held constant during
        fitting (e.g., {"R0": 0.1}). Defaults to an empty dictionary.

    Returns
    -------
    bounds : tuple of list
        A 2-tuple containing lists of lower and
        upper bounds for the parameters.

    """

    # extract the elements from the circuit
    extracted_elements = extract_circuit_elements(circuit)

    # loop through bounds
    lower_bounds, upper_bounds = [], []

    for elem in extracted_elements:

        raw_element = get_element_from_name(elem)

        for i in range(check_and_eval(raw_element).num_params):

            if elem in constants or elem + f'_{i}' in constants:

                continue

            elif raw_element in ['CPE', 'La'] and i == 1:
                upper_bounds.append(1)
                lower_bounds.append(0)
            # The following are for nleis.py

            elif raw_element in ['TPn'] and i == 3:
                upper_bounds.append(0.5)
                lower_bounds.append(-0.5)
            elif raw_element in ['RCn'] and i == 2:
                upper_bounds.append(0.5)
                lower_bounds.append(-0.5)
            elif raw_element in ['TDSn', 'TDPn', 'TDCn', 'RCSQn',
                                 'RCDQn'] and (i == 5):
                upper_bounds.append(np.inf)
                lower_bounds.append(-np.inf)
            elif raw_element in ['TDSn', 'TDPn', 'TDCn', 'RCSQn',
                                 'RCDQn'] and i == 6:
                upper_bounds.append(0.5)
                lower_bounds.append(-0.5)
            elif raw_element in ['RCDn', 'RCSn'] and (i == 4):
                upper_bounds.append(np.inf)
                lower_bounds.append(-np.inf)
            elif raw_element in ['RCDn', 'RCSn'] and i == 5:
                upper_bounds.append(0.5)
                lower_bounds.append(-0.5)
            elif raw_element in ['TLMn'] and (i == 6 or i == 7):
                upper_bounds.append(0.5)
                lower_bounds.append(-0.5)
            elif raw_element in ['TLMSn', 'TLMDn'] and (i == 9 or i == 10):
                upper_bounds.append(0.5)
                lower_bounds.append(-0.5)
            elif raw_element in ['TLMSn', 'TLMDn'] and (i == 8):
                upper_bounds.append(np.inf)
                lower_bounds.append(-np.inf)
            elif raw_element in ['K', 'Kn'] and i == 0:
                upper_bounds.append(np.inf)
                lower_bounds.append(-np.inf)
            elif raw_element in ['RCSQ', 'RCSQn',
                                 'RCDQ', 'RCDQn'] and (i == 2):
                upper_bounds.append(1)
                lower_bounds.append(0)
            else:
                upper_bounds.append(np.inf)
                lower_bounds.append(0)

    bounds = ((lower_bounds), (upper_bounds))

    return bounds

# adopt from impedance.py to support 2nd-NLEIS/NLEIS fitting


def circuit_fit(frequencies, impedances, circuit, initial_guess, constants={},
                bounds=None, weight_by_modulus=False, global_opt=False,
                graph=False,
                **kwargs):
    """ Main function for fitting an equivalent circuit to data.

    By default, this function uses `scipy.optimize.curve_fit
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_
    to fit the equivalent circuit. This function generally works well for
    simple circuits. However, the final results may be sensitive to
    the initial conditions for more complex circuits. In these cases,
    the `scipy.optimize.basinhopping
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html>`_
    global optimization algorithm can be used to attempt a better fit.

    Parameters
    ----------
    frequencies : numpy array
        Frequencies

    impedances : numpy array of dtype 'complex128'
        Impedances

    circuit : string
        String defining the equivalent circuit to be fit

    initial_guess : list of floats
        Initial guesses for the fit parameters

    constants : dictionary, optional
        Parameters and their values to hold constant during fitting
        (e.g. {"RO": 0.1}). Defaults to {}

    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to bounds on all
        parameters of 0 and np.inf, except the CPE alpha
        which has an upper bound of 1

    weight_by_modulus : bool, optional
        Uses the modulus of each data (`|Z|`) as the weighting factor.
        Standard weighting scheme when experimental variances are unavailable.
        Only applicable when global_opt = False

    global_opt : bool, optional
        If global optimization should be used (uses the basinhopping
        algorithm). Defaults to False

    graph : bool, optional
        Whether to use execution graph to process the circuit.
        Defaults to False, which uses eval based code

    kwargs :
        Keyword arguments passed to scipy.optimize.curve_fit or
        scipy.optimize.basinhopping

    Returns
    -------
    p_values : list of floats
        best fit parameters for specified equivalent circuit

    p_errors : list of floats
        one standard deviation error estimates for fit parameters

    Notes
    -----
    Need to do a better job of handling errors in fitting.
    Currently, an error of -1 is returned.

    """

    f = np.array(frequencies, dtype=float)
    Z = np.array(impedances, dtype=complex)

    # set upper and lower bounds on a per-element basis
    if bounds is None:
        bounds = set_default_bounds(circuit, constants=constants)

    cg = CircuitGraph(circuit, constants)

    if not global_opt:
        if 'maxfev' not in kwargs:
            kwargs['maxfev'] = int(1e5)
        if 'ftol' not in kwargs:
            kwargs['ftol'] = 1e-13

        # weighting scheme for fitting
        if weight_by_modulus:
            abs_Z = np.abs(Z)
            kwargs['sigma'] = np.hstack([abs_Z, abs_Z])
        if graph:
            popt, pcov = curve_fit(cg.compute_long, f,
                                   np.hstack([Z.real, Z.imag]),
                                   p0=initial_guess,
                                   bounds=bounds,
                                   **kwargs,
                                   )
        else:
            popt, pcov = curve_fit(wrapCircuit(circuit, constants), f,
                                   np.hstack([Z.real, Z.imag]),
                                   p0=initial_guess, bounds=bounds, **kwargs)

        # Calculate one standard deviation error estimates for fit parameters,
        # defined as the square root of the diagonal of the covariance matrix.
        # https://stackoverflow.com/a/52275674/5144795
        perror = np.sqrt(np.diag(pcov))

    else:
        if 'seed' not in kwargs:
            kwargs['seed'] = 0

        def opt_function(x):
            """ Short function for basinhopping to optimize over.
            We want to minimize the RMSE between the fit and the data.

            Parameters
            ----------
            x : args
                Parameters for optimization.

            Returns
            -------
            function
                Returns a function (RMSE as a function of parameters).
            """
            return rmse(wrapCircuit(circuit, constants)(f, *x),
                        np.hstack([Z.real, Z.imag]))

        def opt_function_graph(x):
            return rmse(cg.compute_long(f, *x), np.hstack([Z.real, Z.imag]))

        class BasinhoppingBounds(object):
            """ Adapted from the basinhopping documetation
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
            """

            def __init__(self, xmin, xmax):
                self.xmin = np.array(xmin)
                self.xmax = np.array(xmax)

            def __call__(self, **kwargs):
                x = kwargs['x_new']
                tmax = bool(np.all(x <= self.xmax))
                tmin = bool(np.all(x >= self.xmin))
                return tmax and tmin

        basinhopping_bounds = BasinhoppingBounds(xmin=bounds[0],
                                                 xmax=bounds[1])
        if graph:
            results = basinhopping(opt_function_graph, x0=initial_guess,
                                   accept_test=basinhopping_bounds, **kwargs)
        else:
            results = basinhopping(opt_function, x0=initial_guess,
                                   accept_test=basinhopping_bounds, **kwargs)
        popt = results.x

        # Calculate perror
        jac = results.lowest_optimization_result['jac'][np.newaxis]
        try:
            # jacobian -> covariance
            # https://stats.stackexchange.com/q/231868
            pcov = inv(np.dot(jac.T, jac)) * opt_function(popt) ** 2
            # covariance -> perror (one standard deviation
            # error estimates for fit parameters)
            perror = np.sqrt(np.diag(pcov))
        except (ValueError, np.linalg.LinAlgError):
            warnings.warn('Failed to compute perror')
            perror = None

    return popt, perror

# adopt from impedance.py to support 2nd-NLEIS/NLEIS fitting


def wrapCircuit(circuit, constants):
    """ wraps function so we can pass the circuit string """
    def wrappedCircuit(frequencies, *parameters):
        """ returns a stacked array of real and imaginary impedance
        components

        Parameters
        ----------
        circuit : string
        constants : dict
        parameters : list of floats
        frequencies : list of floats

        Returns
        -------
        array of floats

        """

        x = eval(buildCircuit(circuit, frequencies, *parameters,
                              constants=constants, eval_string='',
                              index=0)[0],
                 circuit_elements)
        y_real = np.real(x)
        y_imag = np.imag(x)

        return np.hstack([y_real, y_imag])
    return wrappedCircuit

# modified to enable subtraction


def buildCircuit(circuit, frequencies, *parameters,
                 constants=None, eval_string='', index=0):
    """ recursive function that transforms a circuit, parameters, and
    frequencies into a string that can be evaluated

    Parameters
    ----------
    circuit: str
    frequencies: list/tuple/array of floats
    parameters: list/tuple/array of floats
    constants: dict

    Returns
    -------
    eval_string: str
        Python expression for calculating the resulting fit
    index: int
        Tracks parameter index through recursive calling of the function

    """

    parameters = np.array(parameters).tolist()
    frequencies = np.array(frequencies).tolist()
    circuit = circuit.replace(' ', '')

    def parse_circuit(circuit, parallel=False, series=False, difference=False):
        """ Splits a circuit string by either dashes (series) or commas
            (parallel) outside of any paranthesis. Removes any leading 'p('
            or trailing ')' when in parallel mode or 'd(' or trailing ')'
            when in difference mode. This is adapted from impedance.py to
            support subtraction """

        assert (parallel != series
                or series != difference
                or difference != parallel), \
            'Exactly one of parallel or series or difference must be True'

        def count_parens(string):
            return string.count('('), string.count(')')

        if parallel:
            special = ','
            if circuit.endswith(')') and circuit.startswith('p('):
                circuit = circuit[2:-1]
        if difference:
            special = ','
            if circuit.endswith(')') and circuit.startswith('d('):
                circuit = circuit[2:-1]

        if series:
            special = '-'

        split = circuit.split(special)

        result = []
        skipped = []
        for i, sub_str in enumerate(split):
            if i not in skipped:
                if '(' not in sub_str and ')' not in sub_str:
                    result.append(sub_str)
                else:
                    open_parens, closed_parens = count_parens(sub_str)
                    if open_parens == closed_parens:
                        result.append(sub_str)
                    else:
                        uneven = True
                        while i < len(split) - 1 and uneven:
                            sub_str += special + split[i+1]

                            open_parens, closed_parens = count_parens(sub_str)
                            uneven = open_parens != closed_parens

                            i += 1
                            skipped.append(i)
                        result.append(sub_str)
        return result

    parallel = parse_circuit(circuit, parallel=True)
    series = parse_circuit(circuit, series=True)
    difference = parse_circuit(circuit, difference=True)

    if series is not None and len(series) > 1:
        eval_string += "s(["
        split = series
    elif parallel is not None and len(parallel) > 1:
        eval_string += "p(["
        split = parallel

    # added for nleis.py
    elif difference is not None and len(difference) > 1:
        eval_string += "d(["
        split = difference

    elif series == parallel == difference:  # only single element
        split = series

    for i, elem in enumerate(split):
        if ',' in elem or '-' in elem:
            eval_string, index = buildCircuit(elem, frequencies,
                                              *parameters,
                                              constants=constants,
                                              eval_string=eval_string,
                                              index=index)
        else:
            param_string = ""
            raw_elem = get_element_from_name(elem)
            elem_number = check_and_eval(raw_elem).num_params
            param_list = []
            for j in range(elem_number):
                if elem_number > 1:
                    current_elem = elem + '_{}'.format(j)
                else:
                    current_elem = elem

                if current_elem in constants.keys():
                    param_list.append(constants[current_elem])
                else:
                    param_list.append(parameters[index])
                    index += 1

            param_string += str(param_list)
            new = raw_elem + '(' + param_string + ',' + str(frequencies) + ')'
            eval_string += new

        if i == len(split) - 1:
            if len(split) > 1:  # do not add closing brackets if single element
                eval_string += '])'
        else:
            eval_string += ','

    return eval_string, index

# adopt from impedance.py to support 2nd-NLEIS/NLEIS fitting


def calculateCircuitLength(circuit):
    """ Calculates the number of elements in the circuit.

    Parameters
    ----------
    circuit : str
        Circuit string.

    Returns
    -------
    length : int
        Length of circuit.

    """
    length = 0
    if circuit:
        extracted_elements = extract_circuit_elements(circuit)
        for elem in extracted_elements:
            raw_element = get_element_from_name(elem)
            num_params = check_and_eval(raw_element).num_params
            length += num_params
    return length

# modified to enable subtraction


def extract_circuit_elements(circuit):
    """ Extracts circuit elements from a circuit string.

    Parameters
    ----------
    circuit : str
        Circuit string.

    Returns
    -------
    extracted_elements : list
        list of extracted elements.

    """
    p_string = [x for x in circuit if x not in 'p(),-d']
    extracted_elements = []
    current_element = []
    length = len(p_string)
    for i, char in enumerate(p_string):
        if char not in ints:
            current_element.append(char)
        else:
            # min to prevent looking ahead past end of list
            if p_string[min(i+1, length-1)] not in ints:
                current_element.append(char)
                extracted_elements.append(''.join(current_element))
                current_element = []
            else:
                current_element.append(char)
    extracted_elements.append(''.join(current_element))
    return extracted_elements

# Circuit Graph for computation optimization
# Special Thanks to Jake Anderson for the original code


class CircuitGraph:
    '''
    A class to represent a circuit as a directed graph.
    '''
    # regular expression to find parallel and difference blocks
    _parallel_difference_block_expression = re.compile(r'(?:p|d)\([^()]*\)')

    # regular expression to remove whitespace
    _whitespce = re.compile(r"\s+")

    def __init__(self, circuit, constants=None):
        '''
        Initialize the CircuitGraph object.'''
        # remove all whitespace from the circuit string
        self.circuit = self._whitespce.sub("", circuit)
        # parse the circuit string and initialize the graph
        self.parse_circuit()
        # compute the execution order of the graph
        self.execution_order = list(nx.topological_sort(self.graph))
        # initialize the constants dictionary
        self.constants = constants if constants is not None else dict()

    def parse_circuit(self):
        '''
        Parse the circuit string and initialize the graph.
        '''
        # initialize the node counters for each type of block
        self.snum = 1
        self.pnum = 1
        self.dnum = 1
        # initialize the circuit string to be parsed
        parsing_circuit = self.circuit

        # determine all of the base elements, their functions
        # and add them to the graph
        element_name = extract_circuit_elements(parsing_circuit)
        element_func = [
            circuit_elements[get_element_from_name(e)] for e in element_name
        ]
        # graph initialization
        self.graph = nx.DiGraph()
        # add nodes to the graph
        for e, f in zip(element_name, element_func):
            self.graph.add_node(e, Z=f)

        # find unnested parallel and difference blocks
        pd_blocks = self._parallel_difference_block_expression.findall(
            parsing_circuit)

        while len(pd_blocks) > 0:
            # add parallel or difference blocks to the graph
            # unnesting each time around the loop
            for pd in pd_blocks:
                operator = pd[0]
                pd_elem = pd[2:-1].split(",")

                if operator == "p":
                    nnum = self.pnum
                    self.pnum += 1
                elif operator == "d":
                    nnum = self.dnum
                    self.dnum += 1

                node = f"{operator}{nnum}"
                self.graph.add_node(node, Z=circuit_elements[operator])
                for elem in pd_elem:
                    elem = self.add_series_elements(elem)
                    self.graph.add_edge(elem, node)
                parsing_circuit = parsing_circuit.replace(pd, node)

            pd_blocks = self._parallel_difference_block_expression.findall(
                parsing_circuit)

        # pick up any top line series connections
        self.add_series_elements(parsing_circuit)

        # assign layers to the nodes
        for layer, nodes in enumerate(nx.topological_generations(self.graph)):
            for n in nodes:
                self.graph.nodes[n]["layer"] = layer
    # function to add series elements to the graph

    def add_series_elements(self, elem):
        '''
        Add series elements to the graph.
        '''
        selem = elem.split("-")
        if len(selem) > 1:
            node = f"s{self.snum}"
            self.snum += 1
            self.graph.add_node(node, Z=circuit_elements["s"])
            for n in selem:
                self.graph.add_edge(n, node)
            return node

        # if there isn't a series connection in elem just return it unchanged
        return selem[0]

    # function to visualize the graph
    def visualize_graph(self, **kwargs):
        '''
        Visualize the graph.'''
        pos = nx.multipartite_layout(self.graph, subset_key="layer")
        nx.draw_networkx(self.graph, pos=pos, **kwargs)

    # function to compute the impedance of the circuit
    def compute(self, f, *parameters):
        '''
        Compute the impedance of the circuit at the given frequencies.
        '''
        node_results = {}
        pindex = 0
        for node in self.execution_order:
            Zfunc = self.graph.nodes[node]["Z"]
            plist = [
                node_results[pred] for pred in self.graph.predecessors(node)
            ]

            if len(plist) < 1:
                n_params = Zfunc.num_params
                for j in range(n_params):
                    p_name = format_parameter_name(node, j, n_params)
                    if p_name in self.constants:
                        plist.append(self.constants[p_name])
                    else:
                        plist.append(parameters[pindex])
                        pindex += 1
                node_results[node] = Zfunc(plist, f)
            else:
                node_results[node] = Zfunc(plist)

        return np.squeeze(node_results[node])

    # To enable comparision

    def __eq__(self, other):
        '''
        Compare two CircuitGraph objects for equality.
        '''
        if not isinstance(other, CircuitGraph):
            return False
        # Compare the internal graph attributes
        return (self.graph.nodes == other.graph.nodes
                and self.graph.edges == other.graph.edges)

    # To enable direct calling

    def __call__(self, f, *parameters):
        '''
        Compute the impedance of the circuit at the given frequencies.
        '''
        Z = self.compute(f, *parameters)
        return Z

    def compute_long(self, f, *parameters):
        '''
        Compute the impedance of the circuit at the given frequencies.
        And convert it to a long array for curve_fit.
        '''
        Z = self.compute(f, *parameters)
        return np.hstack([Z.real, Z.imag])

    def calculate_circuit_length(self):
        '''
        calculate the number of parameters in the circuit
        '''
        n_params = [
            getattr(Zfunc, "num_params", 0)
            for node, Zfunc in self.graph.nodes(data="Z")
        ]
        return np.sum(n_params)


def format_parameter_name(name, j, n_params):
    '''
    Format the parameter name for the given element.
    '''
    return f"{name}_{j}" if n_params > 1 else f"{name}"
