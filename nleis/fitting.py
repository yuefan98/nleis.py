import numpy as np
from impedance.models.circuits.elements import circuit_elements
from impedance.models.circuits.elements import get_element_from_name
from impedance.models.circuits.fitting import check_and_eval, rmse

# Note: a lot of codes are directly adopted from impedance.py.,
# which is designed to be enable a easy integration in the future,
# but now we are keep them separated to ensure the stable performance

ints = '0123456789'


def mae(a, b):
    '''

    Mean Absolute Error

    Parameters
    ----------
    a : numpy array
        experimental data.
    b : numpy array
        model fit.

    Returns
    -------
    The calculated Mean Absolute Error.

    '''
    return (np.mean(abs(a-b)))


def mape(a, b):
    '''

    Mean Absolute Percentage Error

    Parameters
    ----------
    a : numpy array
        experimental data.
    b : numpy array
        model fit.

    Returns
    -------
    Mean Absolute Percentage Error.

    '''
    return (np.mean(abs(a-b)/abs(a))*100)


def seq_fit_parm(input_dic, target_arr, output_arr):
    '''

    Convert obtained EIS result to a constant of dictionary
    for 2nd-NLEIS analysis using sequential optimization disscussed in [1]
    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    Parameters
    ----------
    input_dic : dictionary
        dictionary of EIS fitting results.
    target_arr : list of string
        a lit of EIS circuit element that need to be converted.
        i.e. ['TDS0','TDS1']
    output_arr : list of string
        a lit of 2nd-NLEIS circuit element that will be converted to.
        i.e. ['TDSn0','TDSn1']

    Raises
    ------
    ValueError
        Raised if the Target Array and Output Array have differnt length.

    Returns
    -------
    output_dic : dictionary
        A dictionary of constant that can be used
        for equential 2nd-NLEIS optimization

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

    This function sets default bounds for optimization.

    set_default_bounds sets bounds of 0 and np.inf for all parameters,
    Exceptions:
    the CPE and La alpha have an upper bound of 1,
    symmetry parameter (ε) for 2nd-NLEIS
    has bounds between -0.5 to 0.5
    curvature parameter (κ) for 2nd-NLEIS
    has bounds between -np.inf to np.inf

    Parameters
    ----------
    circuit : string
        String defining the equivalent circuit to be fit

    constants : dictionary, optional
        Parameters and their values to hold constant during fitting
        (e.g. {"RO": 0.1}). Defaults to {}

    Returns
    -------
    bounds : 2-tuple of array_like
        Lower and upper bounds on parameters.

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
            elif raw_element in ['Tsn'] and i == 4:
                upper_bounds.append(0.5)
                lower_bounds.append(-0.5)
            elif raw_element in ['TPn'] and i == 3:
                upper_bounds.append(0.5)
                lower_bounds.append(-0.5)
            elif raw_element in ['RCn', 'RCOn'] and i == 2:
                upper_bounds.append(0.5)
                lower_bounds.append(-0.5)
            elif raw_element in ['TDSn', 'TDPn', 'TDCn'] and (i == 5):
                upper_bounds.append(np.inf)
                lower_bounds.append(-np.inf)
            elif raw_element in ['TDSn', 'TDPn', 'TDCn'] and i == 6:
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
            elif raw_element in ['TLMSn'] and (i == 9 or i == 10):
                upper_bounds.append(0.5)
                lower_bounds.append(-0.5)
            elif raw_element in ['TLMSn'] and (i == 8):
                upper_bounds.append(np.inf)
                lower_bounds.append(-np.inf)
            else:
                upper_bounds.append(np.inf)
                lower_bounds.append(0)

    bounds = ((lower_bounds), (upper_bounds))

    return bounds

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

# modified to ignore d (difference) for NLEIS circuits
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
    p_string = [x for x in circuit if x not in 'p(),-,d()']
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
