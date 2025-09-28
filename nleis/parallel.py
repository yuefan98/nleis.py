import numpy as np
from joblib import Parallel, delayed
from SALib.sample import sobol
from .validation import cost_max_norm

try:
    from threadpoolctl import threadpool_limits
except Exception:
    # Safe no-op if threadpoolctl isn't installed
    class threadpool_limits:
        def __init__(self, *_, **__): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False


def fit_once(base_model, initial_guess, f, show_results=True, **fit_kwargs):
    """
    Run one fit with initial_guess using the base_model
    Parameters
    ----------
    base_model : instance of circuit for EIS, NLEIS, EISandNLEIS
        The model to be fitted. The initial_guess attribute will be used as
        the starting point for the optimization.
    initial_guess : array_like
        The initial guess for the fitting parameters.
    f : array_like
        Frequencies at which the data is measured.
    show_results : bool, optional
        Whether to print the fitting results. The default is True.
    **fit_kwargs : keyword arguments
        Additional keyword arguments to pass to the fit method of the model
        from CustomCircuit, NLESCustomCircuit, EISandNLEIS.
        Supported keywords are:
        - Z1: array_like, the first set of impedance data to fit.
        - Z2: array_like, the second set of impedance data to fit.
        - impedance: array_like, the impedance data to fit
        (if fitting EIS or NLEIS only).
        - cost_func: callable, the cost function to use for fitting.
        The default is cost_max_norm. It must take two arguments:
        the measured data
        and the fitted data, and return a scalar cost value.
        Other cost functions will be supported in the future.
        - cost: float, the weight of cost
            function when fitting both Z1 and Z2.
            A value greater than 0.5 puts more weight on EIS,
            while a value less than 0.5 puts more weight on 2nd-NLEIS.
            (i.e. cost*cost_func(Z1)+(1-cost)*cost_func(Z2))
            The default is 0.5.
        - max_f : float, optional
            The maximum frequency of interest for 2nd-NLEIS. Default is np.inf.
            Avaliable only with EISandNLEIS and NLESCustomCircuit circuit.
    Returns
    -------
    results : dict
        A dictionary containing the fitting results:
        - 'Status': bool, whether the fit was successful.
        - 'err': str, the error message if the fit failed.
        - 'p0': array_like, the initial guess used for the fit.
    """

    try:
        # Validate inputs
        Z1_data = fit_kwargs.pop("Z1", None)
        Z2_data = fit_kwargs.pop("Z2", None)
        impedance_data = fit_kwargs.pop("impedance", None)

        # Make a private copy (works for both processes & threads)
        model = base_model

        # Set the per-try initial guess
        model.initial_guess = initial_guess

        if fit_kwargs.pop("cost_func", None) is None:
            cost_func = cost_max_norm

        if (Z1_data is not None) and (Z2_data is not None):
            # Avoid BLAS/OpenMP oversubscription inside each worker
            with threadpool_limits(1):
                model.fit(f, Z1_data, Z2_data, **fit_kwargs)
            Z1_fit, Z2_fit = model.predict(f)
            if fit_kwargs.pop("cost", None) is None:
                cost = 0.5

            cost_value = cost_func(Z1_data, Z1_fit) + \
                cost * cost_func(Z2_data, Z2_fit)
        elif impedance_data is not None:
            # Avoid BLAS/OpenMP oversubscription inside each worker

            with threadpool_limits(1):
                model.fit(f, impedance_data, **fit_kwargs)
            Z_fit = model.predict(f)
            cost_value = cost_func(impedance_data, Z_fit)

        else:
            raise ValueError(
                "Either (Z1 and Z2) or "
                "impedance must be provided for fitting.")

        results = {'Status': True, "p0": initial_guess,
                   'p': model.parameters_, "cost": cost_value, "model": model}
        if show_results:
            print(results)
        return results
    except Exception as e:
        results = {"Status": False, "err": repr(e), "p0": initial_guess}
        if show_results:
            print(results)
        return results


def multistart_fit(base_model, f, sampling_method="sobol", num_samples=1024,
                   n_jobs=-1, backend="loky", batch_size="auto", **fit_kwargs):
    """
    Try many initial guesses in parallel. 'guesses' must be an iterable of p0.
    Parameters
    ----------
    base_model : instance of circuit for EIS, NLEIS, EISandNLEIS
        The model to be fitted. The initial_guess attribute will be used as
        the starting point for the optimization.
    f : array_like
        Frequencies at which the data is measured.
    sampling_method : str, optional
        Method to sample the initial guesses. Supported methods are 'sobol'
        and 'random'. The default is 'sobol'.
    num_samples : int, optional
        Number of initial guesses to sample. The default is 1024.
    n_jobs : int, optional
        The number of jobs to run in parallel. The default is -1, which means
        using all available processors.
    backend : str, optional
        The parallelization backend to use. The default is 'loky'.
    batch_size : int or str, optional
        The number of tasks to dispatch at once to each worker. The default is
        'auto'.
    **fit_kwargs : keyword arguments
        Additional keyword arguments to pass to the fit method of the model
        from CustomCircuit, NLESCustomCircuit, EISandNLEIS.
                Supported keywords are:
        - Z1: array_like, the first set of impedance data to fit.
        - Z2: array_like, the second set of impedance data to fit.
        - impedance: array_like, the impedance data to fit
        (if fitting EIS or NLEIS only).
        - cost_func: callable, the cost function to use for fitting.
        The default is cost_max_norm. It must take two arguments:
        the measured data and the fitted data, and return a scalar cost value.
        Other cost functions will be supported in the future.
        - cost: float, the weight of cost
            function when fitting both Z1 and Z2.
            A value greater than 0.5 puts more weight on EIS,
            while a value less than 0.5 puts more weight on 2nd-NLEIS.
            (i.e. cost*cost_func(Z1)+(1-cost)*cost_func(Z2))
            The default is 0.5.
        - max_f : float, optional
            The maximum frequency of interest for 2nd-NLEIS. Default is np.inf.
    Returns
    -------
    best : dict
        The best fitting result.
    results : list of dict
        All fitting results.
    """
    initial_guess = base_model.initial_guess
    if sampling_method == "sobol":
        problem = {
            'num_vars': len(initial_guess),
            'bounds': [[min(guess, 0), 2*guess] for guess in initial_guess]
        }
        initial_guesses = sobol.sample(problem, num_samples)
    elif sampling_method == "random":
        initial_guesses = np.random.uniform(low=0, high=2, size=(
            num_samples, len(initial_guess)))*initial_guess
    else:
        raise ValueError(
            "Unsupported sampling method, only 'sobol' and 'random' "
            "is supported currently.")
    # Adding the sampled initial guess to the inputed initial guess
    initial_guesses = np.vstack([initial_guess, initial_guesses])

    # Create and launch the jobs
    jobs = (
        delayed(fit_once)(base_model, p0, f, **fit_kwargs)
        for p0 in initial_guesses
    )
    # Run the jobs in parallel
    results = Parallel(n_jobs=n_jobs, backend=backend,
                       batch_size=batch_size)(jobs)

    # Filter out failed fits
    fits = [r for r in results if r.get("Status")]
    if not fits:
        raise RuntimeError("All fits failed", results)

    # Select the best fit by cost
    best = min(fits, key=lambda r: r["cost"])

    # Print the best fitting result
    print("Best fitting result: \np0:", best["p0"],
          "\np:", best["p"],
          "\ncost:", best["cost"])
    return best, results
