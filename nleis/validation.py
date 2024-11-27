import matplotlib.pyplot as plt
import numpy as np
from impedance.models.circuits import CustomCircuit
from .nleis import NLEISCustomCircuit
from .visualization import plot_first, plot_second
from tqdm import tqdm


def NL_MM(f, Z, raw_circuit='Kn', initial_guess=[0.01, 1], tol=1e-5,
          plot=False, max_f=10, max_M=20):
    """
    Perform NLEIS fitting using nonlinear measurement models
    with a specified maximum number of elements.

    Parameters:
    f : array-like
        Frequency data
    Z : array-like
        2nd-NLEIS Impedance data
    raw_circuit: str
        Basis function for the measurement model
    initial_guess : list
        Initial guess for model parameters
    tol : float
        Tolerance for cost convergence
        Default: 1e-5
    plot : bool
        Whether to plot the results during the fitting process
        Default = False
    max_f : float
        Maximum frequency cutoff
        Default = 10 Hz
    max_M : int
        Maximum number of elements to try in the measurement model
        Default = 20

    Returns:
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
        confidence interval of the fitted paramer calculated from covariance

    """
    if max_M < 1:
        raise ValueError(
            'The maximum number of elements (max_M) '
            + 'should always be greater than or equal to 1')

    circuit = raw_circuit+'0'
    p0 = initial_guess * max_M
    n = len(initial_guess)
    cost = [0]
    previous_cost = 0
    mask = f < max_f
    f = f[mask]
    Z = Z[mask]
    with tqdm(total=max_M, file=None) as pbar:
        for i in range(max_M):
            M = i + 1
            model = NLEISCustomCircuit(circuit, initial_guess=p0[:n*M])
            model.fit(f, Z, max_f=max_f)
            p0[:n*M] = model.parameters_

            Z_fit = model.predict(f)

            current_cost = cost_max_norm(Z, Z_fit)
            cost.append(current_cost)

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

            if abs(current_cost - previous_cost) < tol:
                print(f'Optimal solution found with M = {M}')

                res_real = (Z.real - Z_fit.real) / Z.real
                res_imag = (Z.imag - Z_fit.imag) / Z.imag

                pbar.update(1)
                pbar.total = M
                pbar.refresh()

                p = model.parameters_
                conf = model.conf_
                return M, cost[1:], p, Z_fit, res_real, res_imag, conf

            previous_cost = current_cost

            circuit += '-' + raw_circuit + str(i+1)
            pbar.update(1)

    raise ValueError(f'The calculation exceeded max_M = {max_M} without '
                     + 'finding an optimal solution. '
                     f'Final cost difference = {abs(cost[-1] - cost[-2]):.2e}.'
                     + ' Please adjust max_M or other relevant parameters.')


def LinMM(f, Z, raw_circuit='K', initial_guess=[0.01, 1],
          plot=False, CI_plot=False,  max_M=20):
    """
    Perform EIS fitting using measurement models
    with a specified maximum number of elements.
    The LinMM function is aiming to reproduce
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
    plot : bool
        Whether to plot the results during the fitting process
        Default = False
    CI_plot : bool
        Whether to plot the 95% confidence interval calculated using
        monte carlo simulation with random sampling of 1000 sample
        using covariance derived standard deviation.
        Default = False
    max_M : int
        Maximum number of elements to try in the measurement model
        Default = 20

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
        confidence interval of the fitted paramer calculated from covariance

    """
    if max_M < 1:
        raise ValueError(
            'The maximum number of elements (max_M) '
            + 'should always be greater than or equal to 1')

    circuit = raw_circuit+'0'
    p0 = initial_guess * max_M
    n = len(initial_guess)
    Z_fit_previous = 0
    p_previous = 0
    conf_previous = 0
    circuit_previous = ''

    with tqdm(total=max_M, file=None) as pbar:
        for i in range(max_M):

            M = i + 1

            model = CustomCircuit(circuit, initial_guess=p0[:n*M])

            model.fit(f, Z)

            Z_fit = model.predict(f)

            p_current = model.parameters_
            conf_current = model.conf_
            conf_intervals = [(p - 2*conf, p + 2*conf)
                              for p, conf in zip(p_current, conf_current)]

            # Check if any parameter's confidence interval includes zero
            # if it includes zero, return results from previous calculation
            if any(lb <= 0 <= ub for lb, ub in conf_intervals):
                print(f'Optimal solution found with M = {M-1}')
                res_real = (Z.real - Z_fit_previous.real) / Z.real
                res_imag = (Z.imag - Z_fit_previous.imag) / Z.imag
                pbar.total = M-1
                pbar.refresh()
                pbar.close()

                if CI_plot:
                    # Monte Carlo simulation for confidence intervals
                    n_simulations = 1000
                    monte_simulation = np.zeros(
                        (n_simulations, len(f)), dtype='complex128')

                    model = CustomCircuit(
                        circuit_previous, initial_guess=p_previous)
                    len_p = len(p_previous)

                    for i in range(n_simulations):
                        # Generate random parameters from a normal distribution
                        p1 = p_previous + \
                            np.random.normal(0, conf_previous, size=len_p)

                        model.initial_guess = p1
                        monte_simulation[i] = model.predict(f)

                    # Calculate confidence intervals
                    # 95.4% confidence (lower)
                    lower_bound = np.percentile(monte_simulation, 2.3, axis=0)
                    # 95.4% confidence (upper)
                    upper_bound = np.percentile(monte_simulation, 97.7, axis=0)

                    fig, ax = plt.subplots(figsize=(5, 5))

                    # plot data and fit
                    plot_first(ax, Z, fmt='-o')
                    plot_first(ax, Z_fit_previous, fmt='-o')

                    # plot the confidence interval
                    plot_first(ax, lower_bound, fmt='r--')
                    plot_first(ax, upper_bound, fmt='r--')

                    plt.legend(['Data', 'Fit', '95% CI'])

                    plt.show()
                M = M-1
                p = p_previous
                Z_fit = Z_fit_previous
                conf = conf_previous

                return (M, p, Z_fit, res_real, res_imag,
                        conf)

            p_previous = model.parameters_
            conf_previous = model.conf_
            Z_fit_previous = Z_fit
            circuit_previous = circuit

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
