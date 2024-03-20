import warnings

import numpy as np
from scipy.linalg import inv
from scipy.optimize import curve_fit, basinhopping
from impedance.models.circuits.elements import circuit_elements, get_element_from_name
from impedance.models.circuits.fitting import check_and_eval,rmse

### Note: a lot of codes are directly adopted from impedance.py. Can be easily perform a full integration in the future, but now we are keep them separated to ensure the stable performance

ints = '0123456789'


def mae(a,b):
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
    return(np.mean(abs(a-b)))

def mape(a,b):
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
    return(np.mean(abs(a-b)/abs(a))*100)

def seq_fit_parm(input_dic,target_arr,output_arr):
    '''
     
    Convert obtained EIS result to a constant of dictionary for 2nd-NLEIS analyssis
    for sequential optimization disscussed in [1]

    Parameters
    ----------
    input_dic : dictionary
        DESCRIPTION.
    target_arr : list of string
        DESCRIPTION.
    output_arr : list of string 
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    output_dic = {}
    n = len(target_arr)
    if n != len(output_arr):
        raise ValueError('Target Array and Output Array Must Have the Same Length ')

    for i in range(0,n):
        for j in range(check_and_eval(target_arr[i][0:-1]).num_params):
            if target_arr[i]+'_'+str(j) not in input_dic:
                continue
            output_dic[output_arr[i]+'_'+str(j)]=input_dic[target_arr[i]+'_'+str(j)]
    return(output_dic)
    
def set_default_bounds(circuit, constants={}):
    """ 
    
    This function sets default bounds for optimization.

    set_default_bounds sets bounds of 0 and np.inf for all parameters,
    except the CPE and La alphas which have an upper bound of 1.

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
            ## The following are for nleis.py
            elif raw_element in ['Tsn'] and i == 4:
                upper_bounds.append(0.5)
                lower_bounds.append(-0.5)
            elif raw_element in ['TPn'] and i == 3:
                upper_bounds.append(0.5)
                lower_bounds.append(-0.5)
            elif raw_element in ['RCn','RCOn'] and i == 2:
                upper_bounds.append(0.5)
                lower_bounds.append(-0.5)
            elif raw_element in ['TDSn','TDPn','TDCn'] and (i == 5 ) :

                upper_bounds.append(np.inf)
                lower_bounds.append(-np.inf)

            elif raw_element in ['TDSn','TDPn','TDCn'] and i == 6 :
                upper_bounds.append(0.5)
                lower_bounds.append(-0.5)

            elif raw_element in ['RCDn','RCSn'] and (i == 4) :
                upper_bounds.append(np.inf)
                lower_bounds.append(-np.inf)

            elif raw_element in ['RCDn','RCSn'] and i == 5 :
                upper_bounds.append(0.5)
                lower_bounds.append(-0.5)
            elif raw_element in ['TLMn'] and (i == 6 or i ==7):
                upper_bounds.append(0.5)
                lower_bounds.append(-0.5)

            elif raw_element in ['TLMSn'] and (i == 9 or i ==10):
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



def circuit_fit(frequencies, impedances, circuit, initial_guess, constants={},
                bounds=None, weight_by_modulus=False, global_opt=False,
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

    if not global_opt:
        if 'maxfev' not in kwargs:
            kwargs['maxfev'] = int(1e5)
        if 'ftol' not in kwargs:
            kwargs['ftol'] = 1e-13

        # weighting scheme for fitting
        if weight_by_modulus:
            abs_Z = np.abs(Z)
            kwargs['sigma'] = np.hstack([abs_Z, abs_Z])

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

    def parse_circuit(circuit, parallel=False, series=False,difference=False):
        """ Splits a circuit string by either dashes (series) or commas
            (parallel) outside of any paranthesis. Removes any leading 'p('
            or trailing ')' when in parallel mode or 'd('or trailing ')' when in difference mode '' """

        assert parallel != series or series != difference or difference != parallel, \
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
        
    ## added for nleis.py
    elif difference is not None and len(difference) > 1:
        eval_string += "d(["
        split = difference
        
    elif series == parallel == difference :  # only single element
        split = series
        
    for i, elem in enumerate(split):
        if ',' in elem or '-' in elem :
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




