import warnings

import numpy as np
from scipy.linalg import inv
from scipy.optimize import curve_fit, basinhopping
from impedance.models.circuits.elements import circuit_elements, get_element_from_name
from impedance.models.circuits.fitting import check_and_eval,rmse
from .fitting import set_default_bounds,buildCircuit,calculateCircuitLength,extract_circuit_elements
from scipy.optimize import minimize
ints = '0123456789'

def data_processing(f,Z1,Z2,max_f=10):
    mask = np.array(Z1.imag)<0
    f = f[mask]
    Z1 = Z1[mask]
    Z2 = Z2[mask]
    mask1 = np.array(f)<max_f
    f2_truncated = f[mask1]
    Z2_truncated = Z2 [mask1]
    return (f,Z1,Z2,f2_truncated,Z2_truncated)

def simul_fit(frequencies, Z1, Z2, circuit_1,circuit_2, edited_circuit, initial_guess, constants_1={},constants_2={},
                bounds = None, opt='max',cost = 0.5,max_f=10,param_norm = True,positive = True,
                **kwargs):

    """ Main function for the simultaneous fitting of EIS and NLEIS edata.

    By default, this function uses `scipy.optimize.curve_fit
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_
    to fit the equivalent circuit. 

    Parameters
    ----------
    frequencies : numpy array
        Frequencies

    Z1 : numpy array of dtype 'complex128'
        EIS
    Z1 : numpy array of dtype 'complex128'
        NLEIS

    circuit_1 : string
        String defining the EIS equivalent circuit to be fit
    circuit_2 : string
        String defining the NLEIS equivalent circuit to be fit

    initial_guess : list of floats
        Initial guesses for the fit parameters

    constants : dictionary, optional
        Parameters and their values to hold constant during fitting
        (e.g. {"RO": 0.1}). Defaults to {}

    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to bounds on all
        parameters of 0 and np.inf, except the CPE alpha
        which has an upper bound of 1

    opt : str, optional
        Default is max normalization. Other normalization will be supported
        in the future 
    cost : float, default = 0.5
        cost function: cost > 0.5 means more weight on EIS while cost < 0.5 means more weight on NLEIS

    max_f: int
        The the maximum frequency of interest for NLEIS
        
    positive : bool, optional
        Defaults to True for only positive nyquist plot
        
    param_norm : bool, optional
         Defaults to True for better convergence
    

    kwargs :
        Keyword arguments passed to scipy.optimize.curve_fit or
        scipy.optimize.basinhopping

    Returns
    -------
    p_values : list of floats
        best fit parameters for EIS and NLEIS data

    p_errors : list of floats
        one standard deviation error estimates for fit parameters

    """
    ###  Todo fix the negtive loglikelihood
  
    ### set upper and lower bounds on a per-element basis

    if bounds is None:
        combined_constant = constants_2.copy()
        combined_constant.update(constants_1)
        bounds = set_default_bounds(edited_circuit, constants=combined_constant)
        ub = np.ones(len(bounds[1]))
    else:
        if param_norm:
            ub = bounds[1]
            bounds = bounds/ub
        else:
            ub = np.ones(len(bounds[1]))
            
    initial_guess = initial_guess/ub


    if positive:
        mask1 = np.array(Z1.imag)<0
        frequencies = frequencies[mask1]
        Z1 = Z1[mask1]
        Z2 = Z2[mask1]
        mask2 = np.array(frequencies)<max_f
        Z2 = Z2[mask2] 
    else:
        mask2 = np.array(frequencies)<max_f
        Z2 = Z2[mask2] 

    Z1stack = np.hstack([Z1.real, Z1.imag])
    Z2stack = np.hstack([Z2.real, Z2.imag])
    Zstack = np.hstack([Z1stack,Z2stack])
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

        popt, pcov = curve_fit(wrapCircuit_simul(circuit_1, constants_1,circuit_2,constants_2,ub,max_f), frequencies,
                           Zstack,
                           p0=initial_guess, bounds=bounds, **kwargs)
    

    # Calculate one standard deviation error estimates for fit parameters,
    # defined as the square root of the diagonal of the covariance matrix.
    # https://stackoverflow.com/a/52275674/5144795
    # and the following for the bounded and normalized case
    # https://stackoverflow.com/questions/14854339/in-scipy-how-and-why-does-curve-fit-calculate-the-covariance-of-the-parameter-es
        perror = np.sqrt(np.diag(ub*pcov*ub.T))

        return popt*ub, perror
    if opt == 'neg':
        ### This method does not provides converge solution at current development
        bounds = tuple(tuple((bounds[0][i], bounds[1][i])) for i in range(len(bounds[0])))

        res = minimize(wrapNeg_log_likelihood(frequencies,Z1,Z2,circuit_1, constants_1,circuit_2,constants_2,ub,max_f,cost = cost), x0=initial_guess,bounds=bounds, **kwargs)
        
        return (res.x*ub,None)
    

def wrapNeg_log_likelihood(frequencies,Z1,Z2,circuit_1, constants_1,circuit_2,constants_2,ub,max_f=10,cost=0.5):
    ''' wraps function for negtive log likelihood optimization'''
    
    def wrappedNeg_log_likelihood(parameters):
        """ returns a stacked array of real and imaginary impedance
        components

        Parameters
        ----------
        frequencies: list of floats
        Z1: EIS data
        Z2: NLEIS data
        circuit_1 : string        
        constants_1 : dict
        circuit_2 : string        
        constants_2 : dict
        ub : list of floats upper bound if bounds are provided
        max_f: int
        parameters : list of floats

        Returns
        -------
        array of floats

        """
        f1 =frequencies
        mask = np.array(frequencies)<max_f
        f2 = frequencies[mask]
        x1,x2 = wrappedImpedance(circuit_1, constants_1,circuit_2,constants_2,f1,f2,parameters*ub)
        # Z1max = max(np.abs(Z1))
        # Z2max = max(np.abs(Z2))
        # log1 = np.log(sum(((Z1.real-x1.real)/Z1max)**2))+np.log(sum(((Z1.imag-x1.imag)/Z1max)**2))
        # log2 = np.log(sum(((Z2.real-x2.real)/Z2max)**2))+np.log(sum(((Z2.imag-x2.imag)/Z2max)**2))
        log1 = np.log(sum(((Z1-x1))**2))
        log2 = np.log(sum(((Z2-x2))**2))
        return(cost*log1+(1-cost)*log2)
    return wrappedNeg_log_likelihood
        
        


def wrapCircuit_simul(circuit_1, constants_1,circuit_2,constants_2,ub,max_f=10):
    """ wraps function so we can pass the circuit string """
    def wrappedCircuit_simul(frequencies, *parameters):
        """ returns a stacked array of real and imaginary impedance
        components

        Parameters
        ----------
        circuit_1 : string        
        constants_1 : dict
        circuit_2 : string        
        constants_2 : dict
        max_f: int
        parameters : list of floats
        frequencies : list of floats

        Returns
        -------
        array of floats

        """
        
        f1 =frequencies
        mask = np.array(frequencies)<max_f
        f2 = frequencies[mask]
        x1,x2 = wrappedImpedance(circuit_1, constants_1,circuit_2,constants_2,f1,f2,parameters*ub)

        y1_real = np.real(x1)
        y1_imag = np.imag(x1)
        y1_stack = np.hstack([y1_real, y1_imag])
        y2_real = np.real(x2)
        y2_imag = np.imag(x2)
        y2_stack = np.hstack([y2_real, y2_imag])

        return np.hstack([y1_stack, y2_stack])
    return wrappedCircuit_simul

def wrappedImpedance(circuit_1, constants_1,circuit_2,constants_2,f1,f2,parameters):
    '''

    Parameters
    ----------
    circuit_1 : string
    constants_1 : dict
    circuit_2 : string
    constants_2 : dict
    f1 : list of floats
    f2 : list of floats
    parameters : list of floats


    Returns
    -------
    Z1 and Z2

    '''

    p1,p2 = individual_parameters(circuit_1,parameters,constants_1,constants_2)
    
    x1 = eval(buildCircuit(circuit_1, f1, *p1,
                          constants=constants_1, eval_string='',
                          index=0)[0],
             circuit_elements)
    x2 = eval(buildCircuit(circuit_2, f2, *p2,
                          constants=constants_2, eval_string='',
                          index=0)[0],
             circuit_elements)
    return(x1,x2)
def individual_parameters(circuit,parameters,constants_1,constants_2):
    '''

    Parameters
    ----------
    circuit : string
        DESCRIPTION.
    parameters : list of floats
        DESCRIPTION.
    constants_1 : dict
        DESCRIPTION.
    constants_2 : dict
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    '''

    if circuit == '':
        return [],[]
    parameters = list(parameters)
    elements_1 = extract_circuit_elements(circuit)
    p1 = []
    p2 = []
    index = 0
    for elem in elements_1:
        
        raw_elem = get_element_from_name(elem)
        elem_number_1 = check_and_eval(raw_elem).num_params
        if elem[0]=='T' or elem[0:2] =='RC' : ### this might be improvable, but depends on how we want to define the name
            ## check for nonlinear element
            elem_number_2 = check_and_eval(raw_elem+'n').num_params
            
        else: 
            elem_number_2 = elem_number_1
        
        for j in range(elem_number_2):
            if elem_number_1 > 1:
                if j<elem_number_1:
                    current_elem_1 = elem + '_{}'.format(j)
                else:
                    current_elem_1 = None
                len_elem = len(raw_elem)    
                current_elem_2 = elem[0:len_elem]+'n'+elem[len_elem:] + '_{}'.format(j)
            else:
                current_elem_1 = elem
                current_elem_2 = elem
            if current_elem_1 in constants_1.keys()  :
                continue
            elif current_elem_2 in constants_2.keys() and current_elem_1 not in constants_1.keys():
                continue
            else:
                if elem_number_1 ==1:    
                    p1.append(parameters[index])
                    
                elif elem_number_1>1 and j<elem_number_1: 
                    p1.append(parameters[index])
                    p2.append(parameters[index])
                else:
                    p2.append(parameters[index])
                
                index += 1

    return p1,p2









