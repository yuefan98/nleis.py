import matplotlib.pyplot as plt
import numpy as np
from .nleis import NLEISCustomCircuit
from .visualization import plot_first, plot_second
from tqdm import tqdm
import warnings

# set random seed for reproducibility
np.random.seed(0)


def MM(f, Z, raw_circuit='Kn', initial_guess=[0.01, 1], method='cost',
       max_f=np.inf, max_M=20, tol=1e-5, k=1,
       graph=True, plot=False, CI_plot=False):
    '''
    Wrapper function to perform measurement models fitting with either cost
    or confidence interval method.

    Parameters
    -----------
    f : array-like
        Frequency data
    Z : array-like
        Impedance data
    raw_circuit: str
        Basis function for the measurement model
    initial_guess : list
        Initial guess for the basis model parameters
    method : str
        The method to use for fitting. Either 'cost' or 'conf'
        Default: 'conf', which reproduces
        the well-known measurement model program created by Mark Orazem. [1]
        Convergence is not guaranteed
        with the 'conf' method.
        When 'conf' method fails, it is recommended to use the 'cost' method,
        which is developed us and can be found in ref [2].

    max_f : float
        Maximum frequency cutoff
        Default = inf
    max_M : int
        Maximum number of elements to try in the measurement model
        Default = 20
    tol : float
        Tolerance for cost convergence
        Default: 1e-5
    k : int
        Harmonic number for the measurement model to ensure the correct ploting
        Default = 1 for EIS.
        k = 2 for 2nd-NLEIS
    graph : bool
        whether to use graph-based or eval()-based calculation.
        Default = True.

    plot : bool
        Whether to plot the results during the fitting process
        Default = False
    CI_plot : bool
        Whether to plot the 95% confidence interval calculated using
        monte carlo simulation with random sampling of 1000 sample
        using covariance derived standard deviation.
        Default = False

    Returns
    --------
    M : int
        The optimal number of elements determined by the fitting
    p : list
        Final optimized parameters
    conf : array
        confidence interval of the fitted parameter calculated from covariance
    Z_fit : array
        Fitted impedance values
    res_real : array
        Residuals for the real part of the impedance
    res_imag : array
        Residuals for the imaginary part of the impedance
    cost : list
        List of cost values for each iteration.
        Only avaliable when method is 'cost'

    Note:
    -----
    When method = 'cost': a total of 7 outputs are returned.
    When method = 'conf': a total of 6 outputs are returned.

    [1] Orazem, M.E. Measurement model for analysis of
        electrochemical impedance data.
        J Solid State Electrochem 28, 1273–1289 (2024).
        https://doi.org/10.1007/s10008-023-05755-9

    [2] Ji et al, Measurement Model Validation of Second-Harmonic
        Nonlinear Electrochemical Impedance Spectroscopy,
        2025 J. Electrochem. Soc. 172 103506.
        https://iopscience.iop.org/article/10.1149/1945-7111/ae1064/meta

    '''

    if method == 'cost':
        return MM_cost(f, Z, raw_circuit, initial_guess, max_f, max_M, tol, k,
                       graph, plot)
    elif method == 'conf':
        return MM_conf(f, Z, raw_circuit, initial_guess, max_f, max_M, k,
                       graph, plot, CI_plot)
    else:
        raise ValueError('The method should be either cost or conf')


def MM_cost(f, Z, raw_circuit='Kn', initial_guess=[0.01, 1],
            max_f=np.inf, max_M=20, tol=1e-5, k=1, graph=True,
            plot=False):
    """
    Perform NLEIS fitting using nonlinear measurement models
    with a specified maximum number of elements.

    Parameters
    -----------
    f : array-like
        Frequency data
    Z : array-like
        Impedance data
    raw_circuit: str
        Basis function for the measurement model
    initial_guess : list
        Initial guess for model parameters
    max_f : float
        Maximum frequency cutoff
        Default = inf
    max_M : int
        Maximum number of elements to try in the measurement model
        Default = 20
    tol : float
        Tolerance for cost convergence
        Default: 1e-5
    k : int
        Harmonic number for the measurement model to ensure the correct ploting
        Default = 1 for EIS.
        k = 2 for 2nd-NLEIS
    graph : bool
        whether to use graph-based or eval()-based calculation.
        Default = True.
    plot : bool
        Whether to plot the results during the fitting process
        Default = False

    Returns
    --------
    M : int
        The number of elements used in the circuit model
    p : list
        Final optimized parameters
    conf : array
        confidence interval of the fitted parameter calculated from covariance
    Z_fit : array
        Fitted impedance values
    res_real : array
        Residuals for the real part of the impedance
    res_imag : array
        Residuals for the imaginary part of the impedance
    cost : list
        List of cost values for each iteration

    """
    if not isinstance(max_M, int):
        raise TypeError(
            'The maximum number of elements (max_M) should be an integer')
    if max_M < 1:
        raise ValueError(
            'The maximum number of elements (max_M) '
            + 'should always be greater than or equal to 1')
    # Initialize the circuit model and initial guess
    circuit = raw_circuit+'0'
    p0 = initial_guess * max_M
    n = len(initial_guess)
    # Initialize the variables
    cost = [0]
    previous_cost = 0
    Z_fit_previous = 0
    # Mask the data
    mask = f < max_f
    f = f[mask]
    Z = Z[mask]

    # initialize the model
    model = NLEISCustomCircuit(graph=graph)
    # Main loop to fit the model
    with tqdm(total=max_M, file=None) as pbar:
        for i in range(max_M):
            # perform MM fitting with M elements
            M = i + 1

            # update circuit and initial guess for the model
            model.circuit = circuit
            model.initial_guess = p0[:n*M]

            # Fit the model
            model.fit(f, Z, max_f=max_f)

            # update the initial guess
            p0[:n*M] = model.parameters_

            # Predict the impedance
            Z_fit = model.predict(f, max_f=max_f)

            # Compute the cost
            current_cost = cost_max_norm(Z, Z_fit)
            cost.append(current_cost)

            # Check for convergence
            # if abs(current_cost - previous_cost) < tol:
            if (abs(current_cost - previous_cost) < tol or
                    (current_cost > previous_cost and i > 1)):
                # If the cost is less than the tolerance, return the results

                if M == 1:

                    # generate plot
                    if plot:
                        cost_plot(Z, Z_fit, current_cost, M, k)
                    print(f'Optimal solution found with M = {M}')

                    abs_Z = abs(Z)
                    res_real = (Z.real - Z_fit.real) / abs_Z
                    res_imag = (Z.imag - Z_fit.imag) / abs_Z

                    pbar.update(1)
                    pbar.total = M
                    pbar.refresh()

                    p = model.parameters_
                    conf = model.conf_
                    return M, p, conf, Z_fit, res_real, res_imag, cost[1:]
                else:
                    print(f'Optimal solution found with M = {M-1}')

                    abs_Z = abs(Z)
                    res_real = (Z.real - Z_fit_previous.real) / abs_Z
                    res_imag = (Z.imag - Z_fit_previous.imag) / abs_Z

                    pbar.total = M-1
                    pbar.refresh()

                    return (M-1, p, conf, Z_fit_previous, res_real,
                            res_imag, cost[1:-1])

            # generate plot for each iteration
            if plot:
                cost_plot(Z, Z_fit, current_cost, M, k)

            p = model.parameters_
            conf = model.conf_
            Z_fit_previous = Z_fit
            previous_cost = current_cost

            circuit += '-' + raw_circuit + str(i+1)
            pbar.update(1)

    raise ValueError(f'Exceeded max_M without finding an optimal solution. '
                     f'Final cost difference: {abs(cost[-1] - cost[-2]):.2e}. '
                     'Please adjust max_M or other parameters.')


def cost_plot(Z, Z_fit, cost, M, k):
    '''
    Generate plot for the cost function method

    Parameters
    -----------
    Z : array-like,dtype=complex128
        Impedance data
    Z_fit : array-like,dtype=complex128
        Fitted impedance values
    cost : float
        Cost value
    M : int
        Number of elements used in the circuit model
    k : int
        Harmonic number for the measurement model to ensure the correct ploting
        Default = 1 for EIS.
        k = 2 for 2nd-NLEIS

    Returns
    --------
    None
    '''

    fig, ax = plt.subplots(figsize=(5, 5))
    if k == 1:
        plot = plot_first
    elif k == 2:
        plot = plot_second
    else:
        warnings.warn('The harmonic number k should be either 1 or 2. ' +
                      'The default value of 1 will be used, which will give ' +
                      'units and legend corresponding to EIS.')

        plot = plot_first

    plot(ax, Z)
    plot(ax, Z_fit)
    ax.text(0.75, 0.2, '# elements = '
            + str(M),
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes, size=12)
    ax.text(0.75, 0.1, '$COST = $'
            + '{:0.2e}'.format(cost),
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes, size=12)
    plt.legend(['Data', 'Fit'])
    plt.show()
    return ()


def MM_conf(f, Z, raw_circuit='K', initial_guess=[0.01, 1],
            max_f=np.inf, max_M=20, k=1, graph=True,
            plot=False, CI_plot=False):
    """
    Perform impedance fitting using measurement models
    with a specified maximum number of elements.
    The MM_conf function is aiming to reproduce
    the well-known measurement model program created by Mark Orazem, [1]
    aiming to provide an easy accessible python API to do the same analysis.

    [1] Orazem, M.E. Measurement model for analysis of
    electrochemical impedance data.
    J Solid State Electrochem 28, 1273–1289 (2024).
    https://doi.org/10.1007/s10008-023-05755-9

    Parameters
    -----------
    f : array-like
        Frequency data
    Z : array-like
        EIS data
    raw_circuit: str
        Basis function for the measurement model
    initial_guess : list
        Initial guess for model parameters
    max_f : float
        Maximum frequency cutoff
        Default = inf
    max_M : int
        Maximum number of elements to try in the measurement model
        Default = 20
    k : int
        Harmonic number for the measurement model to ensure the correct ploting
        Default = 1 for EIS.
        k = 2 for 2nd-NLEIS
    plot : bool
        Whether to plot the results during the fitting process
        Default = False
    CI_plot : bool
        Whether to plot the 95% confidence interval calculated using
        monte carlo simulation with random sampling of 1000 sample
        using covariance derived standard deviation.
        Default = False

    Returns
    --------
    M : int
        The number of elements used in the circuit model
    p : list
        Final optimized parameters
    conf : array
        confidence interval of the fitted parameter calculated from covariance
    Z_fit : array
        Fitted impedance values
    res_real : array
        Residuals for the real part of the impedance
    res_imag : array
        Residuals for the imaginary part of the impedance

    """
    if not isinstance(max_M, int):
        raise TypeError(
            'The maximum number of elements (max_M) should be an integer')
    if max_M < 1:
        raise ValueError(
            'The maximum number of elements (max_M) '
            + 'should always be greater than or equal to 1')

    # Initialize the circuit model and initial guess
    circuit = raw_circuit+'0'
    p0 = initial_guess * max_M
    n = len(initial_guess)
    # Mask the data
    mask = f < max_f
    f = f[mask]
    Z = Z[mask]

    # Initialize the previous values
    Z_fit_previous = 0
    p_previous = 0
    conf_previous = 0
    circuit_previous = ''

    # initialize the model
    model = NLEISCustomCircuit(graph=graph)

    with tqdm(total=max_M, file=None) as pbar:
        for i in range(max_M):
            # perform MM fitting with M elements
            M = i + 1
            # update circuit and initial guess for the model
            model.circuit = circuit
            model.initial_guess = p0[:n*M]
            # Fit the model
            model.fit(f, Z, max_f=max_f)

            # update the initial guess
            p0[:n*M] = model.parameters_

            # calculate the impedance
            Z_fit = model.predict(f, max_f=max_f)

            # calculate confidence intervals
            p_current = model.parameters_
            conf_current = model.conf_
            conf_intervals = [(p - 2*conf, p + 2*conf)
                              for p, conf in zip(p_current, conf_current)]

            # Check if any parameter's confidence interval includes zero
            # if it includes zero, return results from previous calculation
            if any(lb <= 0 <= ub for lb, ub in conf_intervals):
                if M-1 == 0:
                    raise ValueError("'conf' method failed to find a solution"
                                     + ' with CI does not include zero with '
                                     + 'single basis function.'
                                     + " Please try 'cost' method.")

                print(f'Optimal solution found with M = {M-1}')

                abs_Z = abs(Z)
                res_real = (Z.real - Z_fit_previous.real) / abs_Z
                res_imag = (Z.imag - Z_fit_previous.imag) / abs_Z
                pbar.total = M-1
                pbar.refresh()
                pbar.close()

                if CI_plot:
                    # Perform Monte Carlo simulation and generate plot
                    _, _ = CI_MonteCarlo(f, Z, circuit=circuit_previous,
                                         p=p_previous,
                                         conf=conf_previous, max_f=max_f,
                                         k=k,
                                         graph=graph, plot=True)

                # return the results
                M = M-1
                p = p_previous
                Z_fit = Z_fit_previous
                conf = conf_previous

                return (M, p, conf, Z_fit, res_real, res_imag)

            # Update the previous values
            p_previous = model.parameters_
            conf_previous = model.conf_
            Z_fit_previous = Z_fit
            circuit_previous = circuit

            # generate plot for each iteration
            if plot:
                if k == 1:
                    plot_func = plot_first
                elif k == 2:
                    plot_func = plot_second
                else:
                    warnings.warn('The harmonic number k should be ' +
                                  'either 1 or 2. '
                                  'The default value of 1 will be used, ' +
                                  'which will give units and legend ' +
                                  'corresponding to EIS.')
                    plot_func = plot_first

                _, ax = plt.subplots(figsize=(5, 5))
                plot_func(ax, Z)
                plot_func(ax, Z_fit)
                ax.text(0.75, 0.1, '# elements = '
                        + str(i+1),
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes, size=12)
                plt.legend(['Data', 'Fit'])
                plt.show()

            circuit += '-' + raw_circuit + str(i+1)
            pbar.update(1)

    raise ValueError(f'The calculation exceeded max_M = {max_M} without '
                     + 'finding an optimal solution. '
                     'Please adjust max_M or other relevant parameters.')


def CI_MonteCarlo(f, Z, circuit='', p=[], conf=[], max_f=np.inf,
                  n_simulations=1000, k=1,  graph=True,  plot=False):
    '''
    Perform Monte Carlo simulation to calculate the 95% confidence interval
    of the the impedance spectrum using the confidence interval of
    the parameters calculated using covariance.

    Parameters
    -----------
    f : array-like
        Frequency data
    Z : array-like, dtype=complex128
        Impedance data
    circuit : str
        Circuit for the final measurement model
    p : list
        Fitted measurement model parameters
    conf : list
        Confidence interval of the fitted parameters
    max_f : float
        Maximum frequency cutoff
        Default = inf
    n_simulations : int
        Number of simulations to perform for the Monte Carlo simulation
        Default = 1000
    k : int
        Harmonic number for the measurement model to ensure the correct ploting
        Default = 1 for EIS.
        k = 2 for 2nd-NLEIS
    graph : bool
        whether to use graph-based or eval()-based calculation.
        Default = True.
        To use graph-based calculation, set to True.
    plot : bool
        Whether to plot the final confidence interval results
        Default = False

    Returns
    --------
    lower_bound : array
        Lower bound of the 95% confidence interval
    upper_bound : array
        Upper bound of the 95% confidence interval
        '''

    # initialize monte carlo simulation array and model
    monte_simulation = np.zeros(
        (n_simulations, len(f)), dtype='complex128')

    model = NLEISCustomCircuit(graph=graph)
    model.circuit = circuit
    model.parameters_ = p

    Z_fit = model.predict(f, max_f=max_f)

    len_p = len(p)

    # Perform Monte Carlo simulation
    for i in range(n_simulations):
        # Generate random parameters from a normal distribution
        p1 = p + \
            np.random.normal(0, conf, size=len_p)

        model.parameters_ = p1
        monte_simulation[i] = model.predict(f, max_f=max_f)

    # Calculate confidence intervals

    # Zr_low = np.percentile(monte_simulation.real, 2.5, axis=0)
    # Zr_high = np.percentile(monte_simulation.real, 97.5, axis=0)
    # Zi_low = np.percentile(monte_simulation.imag, 2.5, axis=0)
    # Zi_high = np.percentile(monte_simulation.imag, 97.5, axis=0)
    # lower_bound = Zr_low + 1j*Zi_low
    # upper_bound = Zr_high + 1j*Zi_high

    # # 95% confidence (lower)
    # lower_bound = np.percentile(monte_simulation, 2.5, axis=0)
    # # 95% confidence (upper)
    # upper_bound = np.percentile(monte_simulation, 97.5, axis=0)

    warnings.warn('The confidence interval is calculated based on normal '
                  'projection of the 1000 Monte Carlo simulations '
                  'around the mean curve at each frequency, which converts '
                  'the 2D ellipses into 1D intervals orthogonal '
                  'to the mean curve. '
                  'So that a confidence interval band can be obtained. '
                  'This approach may lead to slight overestimation '
                  'of the confidence intervals compared '
                  'to the true 2D ellipses in the Nyquist plot.')
    # Compute the 95% confidence interval bands based on normal projection
    # Approximate tangent of mean curve at each frequency
    n_freq = len(f)
    t = np.empty(n_freq, dtype=np.complex128)
    mean_mc = monte_simulation.mean(axis=0)
    for i in range(n_freq):
        if i == 0:
            t[i] = mean_mc[1] - mean_mc[0]
        elif i == n_freq - 1:
            t[i] = mean_mc[-1] - mean_mc[-2]
        else:
            t[i] = mean_mc[i + 1] - mean_mc[i - 1]

    # Compute unit normal vectors in the Nyquist plane
    # Treat complex z = x + jy as (x, y)
    nx = -t.imag
    ny = t.real
    norm = np.hypot(nx, ny)

    # Guard against zero tangent (flat or single-point)
    norm[norm == 0] = 1.0

    nx /= norm
    ny /= norm
    n_complex = nx + 1j * ny  # unit normal as complex

    lower_bound = np.empty(n_freq, dtype=np.complex128)
    upper_bound = np.empty(n_freq, dtype=np.complex128)

    for i in range(n_freq):
        # Deviations from mean
        d = monte_simulation[:, i] - mean_mc[i]

        # Signed projection onto normal
        s = d.real * nx[i] + d.imag * ny[i]

        # Quantiles of signed distance along normal
        s_low = np.percentile(s, 2.5)
        s_high = np.percentile(s, 97.5)

        # Map back to complex plane
        lower_bound[i] = mean_mc[i] + s_low * n_complex[i]
        upper_bound[i] = mean_mc[i] + s_high * n_complex[i]

    if plot:
        # Plot the results
        _, ax = plt.subplots(figsize=(5, 5))
        if k == 1:
            plot_func = plot_first
        elif k == 2:
            plot_func = plot_second
        else:
            warnings.warn('The harmonic number k should be ' +
                          'either 1 or 2. '
                          'The default value of 1 will be used, ' +
                          'which will give units and legend ' +
                          'corresponding to EIS.')
            plot_func = plot_first

        # plot data and fit
        plot_func(ax, Z, fmt='-o')
        plot_func(ax, Z_fit, fmt='-o')

        # plot the confidence interval
        plot_func(ax, lower_bound, fmt='r--')
        plot_func(ax, upper_bound, fmt='r--')

        plt.legend(['Data', 'Fit', '95% CI'])

        plt.show()

    return lower_bound, upper_bound


def cost_max_norm(data, model):
    """
    Compute the maximum normalized cost function
    between the data and the model.

    This cost function normalizes both the data and
    the model by the maximum absolute value
    in the data. The cost is computed
    as the sum of the squared differences between the real
    and imaginary parts of the normalized data and model.

    Parameters
    -----------
    data : array-like
        Complex impedance. This array should contain complex numbers,
        and the cost function will
        compare both the real and imaginary components.

    model : array-like
        Complex impedance . This array should be of the same shape
        as `data`, containing complex numbers.
        The cost function will compare both the real
        and imaginary components.

    Returns
    --------
    float
        The sum of the squared differences
        between the real and imaginary parts of the
        normalized data and model. A lower value indicates a better fit.
    """

    Max = max(abs(data))
    data = data/Max
    model = model/Max
    sum1 = np.sum((data.real-model.real)**2)
    sum2 = np.sum((data.imag-model.imag)**2)
    return (sum1+sum2)
