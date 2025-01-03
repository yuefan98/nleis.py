import matplotlib.pyplot as plt
import numpy as np
from .nleis import NLEISCustomCircuit
from .visualization import plot_first, plot_second
from tqdm import tqdm


def MM(f, Z, raw_circuit='K', initial_guess=[0.01, 1], method='cost',
       max_f=np.inf, max_M=20, tol=1e-5, plot=False, CI_plot=False):
    '''
    Wrapper function to perform measurement models fitting with either cost
    or confidence interval method.

    Parameters:
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
    plot : bool
        Whether to plot the results during the fitting process
        Default = False
    CI_plot : bool
        Whether to plot the 95% confidence interval calculated using
        monte carlo simulation with random sampling of 1000 sample
        using covariance derived standard deviation.
        Default = False

    Returns:
    --------
    M : int
        The optimal number of elements determined by the fitting
    cost : list
        List of cost values for each iteration.
        Only avaliable when method is 'cost'
    p : list
        Final optimized parameters
    Z_fit : array
        Fitted impedance values
    res_real : array
        Residuals for the real part of the impedance
    res_imag : array
        Residuals for the imaginary part of the impedance
    conf : array
        confidence interval of the fitted parameter calculated from covariance

    Note:
    -----
    When method = 'cost': a total of 7 outputs are returned.
    When method = 'conf': a total of 6 outputs are returned.

    [1] Orazem, M.E. Measurement model for analysis of
    electrochemical impedance data.
    J Solid State Electrochem 28, 1273–1289 (2024).
    https://doi.org/10.1007/s10008-023-05755-9

    [2] Y. Ji, A. H. Shih D. T. Schwartz

    '''

    if method == 'cost':
        return MM_cost(f, Z, raw_circuit, initial_guess, max_f, max_M, tol,
                       plot)
    elif method == 'conf':
        return MM_conf(f, Z, raw_circuit, initial_guess, max_f, max_M,
                       plot, CI_plot)
    else:
        raise ValueError('The method should be either cost or conf')


def MM_cost(f, Z, raw_circuit='Kn', initial_guess=[0.01, 1],
            max_f=np.inf, max_M=20, tol=1e-5,
            plot=False):
    """
    Perform NLEIS fitting using nonlinear measurement models
    with a specified maximum number of elements.

    Parameters:
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
    plot : bool
        Whether to plot the results during the fitting process
        Default = False

    Returns:
    --------
    M : int
        The number of elements used in the circuit model
    cost : list
        List of cost values for each iteration
    p : list
        Final optimized parameters
    Z_fit : array
        Fitted impedance values
    res_real : array
        Residuals for the real part of the impedance
    res_imag : array
        Residuals for the imaginary part of the impedance
    conf : array
        confidence interval of the fitted parameter calculated from covariance

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
    # Initialize the cost
    cost = [0]
    previous_cost = 0
    # Mask the data
    mask = f < max_f
    f = f[mask]
    Z = Z[mask]
    # Main loop to fit the model
    with tqdm(total=max_M, file=None) as pbar:
        for i in range(max_M):
            # perform MM fitting with M elements
            M = i + 1
            model = NLEISCustomCircuit(circuit, initial_guess=p0[:n*M])
            model.fit(f, Z, max_f=max_f)
            p0[:n*M] = model.parameters_

            Z_fit = model.predict(f, max_f=max_f)

            # Compute the cost
            current_cost = cost_max_norm(Z, Z_fit)
            cost.append(current_cost)

            # generate plot for each iteration
            if plot:
                fig, ax = plt.subplots(figsize=(5, 5))
                plot_second(ax, Z)
                plot_second(ax, Z_fit)
                ax.text(0.75, 0.2, '# elements = '
                        + str(i+1),
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes, size=12)
                ax.text(0.75, 0.1, '$COST = $'
                        + '{:0.2e}'.format(current_cost),
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes, size=12)
                plt.legend(['Data', 'Fit'])
                plt.show()

            # Check for convergence
            if abs(current_cost - previous_cost) < tol:
                # If the cost is less than the tolerance, return the results
                print(f'Optimal solution found with M = {M}')

                res_real = (Z.real - Z_fit.real) / Z.real
                res_imag = (Z.imag - Z_fit.imag) / Z.imag

                pbar.update(1)
                pbar.total = M
                pbar.refresh()

                p = model.parameters_
                conf = model.conf_
                return M, cost[1:], p, Z_fit, res_real, res_imag, conf

            # Update the cost and the circuit model
            previous_cost = current_cost

            circuit += '-' + raw_circuit + str(i+1)
            pbar.update(1)

    raise ValueError(f'Exceeded max_M without finding an optimal solution. '
                     f'Final cost difference: {abs(cost[-1] - cost[-2]):.2e}. '
                     'Please adjust max_M or other parameters.')


def MM_conf(f, Z, raw_circuit='K', initial_guess=[0.01, 1],
            max_f=np.inf, max_M=20,
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

    Parameters:
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
    plot : bool
        Whether to plot the results during the fitting process
        Default = False
    CI_plot : bool
        Whether to plot the 95% confidence interval calculated using
        monte carlo simulation with random sampling of 1000 sample
        using covariance derived standard deviation.
        Default = False

    Returns:
    M : int
        The number of elements used in the circuit model
    p : list
        Final optimized parameters
    Z_fit : array
        Fitted impedance values
    res_real : array
        Residuals for the real part of the impedance
    res_imag : array
        Residuals for the imaginary part of the impedance
    conf : array
        confidence interval of the fitted parameter calculated from covariance

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

    with tqdm(total=max_M, file=None) as pbar:
        for i in range(max_M):
            # perform MM fitting with M elements
            M = i + 1

            model = NLEISCustomCircuit(circuit, initial_guess=p0[:n*M])

            model.fit(f, Z, max_f=max_f)

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

                res_real = (Z.real - Z_fit_previous.real) / Z.real
                res_imag = (Z.imag - Z_fit_previous.imag) / Z.imag
                pbar.total = M-1
                pbar.refresh()
                pbar.close()

                if CI_plot:
                    # Monte Carlo simulation for confidence intervals
                    # with random sampling of 1000 samples
                    n_simulations = 1000

                    # initialize monte carlo simulation array and model
                    monte_simulation = np.zeros(
                        (n_simulations, len(f)), dtype='complex128')

                    model = NLEISCustomCircuit(
                        circuit_previous, initial_guess=p_previous)
                    len_p = len(p_previous)

                    # Perform Monte Carlo simulation
                    for i in range(n_simulations):
                        # Generate random parameters from a normal distribution
                        p1 = p_previous + \
                            np.random.normal(0, conf_previous, size=len_p)

                        model.initial_guess = p1
                        monte_simulation[i] = model.predict(f, max_f=max_f)

                    # Calculate confidence intervals
                    # 95% confidence (lower)
                    lower_bound = np.percentile(monte_simulation, 2.5, axis=0)
                    # 95% confidence (upper)
                    upper_bound = np.percentile(monte_simulation, 97.5, axis=0)

                    # Plot the results
                    fig, ax = plt.subplots(figsize=(5, 5))

                    # plot data and fit
                    plot_first(ax, Z, fmt='-o')
                    plot_first(ax, Z_fit_previous, fmt='-o')

                    # plot the confidence interval
                    plot_first(ax, lower_bound, fmt='r--')
                    plot_first(ax, upper_bound, fmt='r--')

                    plt.legend(['Data', 'Fit', '95% CI'])

                    plt.show()

                # return the results
                M = M-1
                p = p_previous
                Z_fit = Z_fit_previous
                conf = conf_previous

                return (M, p, Z_fit, res_real, res_imag,
                        conf)

            # Update the previous values
            p_previous = model.parameters_
            conf_previous = model.conf_
            Z_fit_previous = Z_fit
            circuit_previous = circuit

            # generate plot for each iteration
            if plot:
                fig, ax = plt.subplots(figsize=(5, 5))
                plot_first(ax, Z)
                plot_first(ax, Z_fit)
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


def cost_max_norm(data, model):
    """
    Compute the maximum normalized cost function
    between the data and the model.

    This cost function normalizes both the data and
    the model by the maximum absolute value
    in the data. The cost is computed
    as the sum of the squared differences between the real
    and imaginary parts of the normalized data and model.

    Parameters:
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

    Returns:
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
