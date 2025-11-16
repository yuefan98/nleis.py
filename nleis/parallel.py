import numpy as np
from joblib import Parallel, delayed
from SALib.sample import sobol
from .validation import cost_max_norm
import traceback


try:
    from threadpoolctl import threadpool_limits
except Exception:
    # Safe no-op if threadpoolctl isn't installed
    class threadpool_limits:
        def __init__(self, *_, **__): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False


def fit_once(base_model, initial_guess, f, run_idx=None, show_results=True,
             **fit_kwargs):
    """
    Run one fit with initial_guess using the base_model

    Parameters
    ----------

    base_model : instance of circuit for EIS, NLEIS, or EISandNLEIS

        The model to be fitted. The initial_guess attribute will be used as
        the starting point for the optimization.

    initial_guess : array_like

        The initial guess for the fitting parameters.

    f : array_like

        Frequencies at which the data is measured.

    run_idx : int or None, optional

        Optional index identifying runs (e.g., from enumerate).

    show_results : bool, optional

        Whether to print the fitting results. The default is True.

    **fit_kwargs : keyword arguments

        Additional keyword arguments to pass to the fit method of the model
        from CustomCircuit, NLESCustomCircuit, or EISandNLEIS.

        Supported keywords are:

        - Z1: array_like. The EIS data to fit.

          For simultaneous EIS and 2nd-NLEIS fitting.

        - Z2: array_like. The 2nd-NLEIS data to fit.

          For simultaneous EIS and 2nd-NLEIS fitting.

        - impedance: array_like. The impedance data to fit.

          For EIS or NLEIS only fitting.

        - cost_func: callable. The default is cost_max_norm

          The cost function to use for fitting.
          It takes two arguments: the measured data and the fitted data,
          and returns a scalar cost value.
          Other cost functions will be supported in the future.

        - cost: float. The default is 0.5.

          The weight of the cost function when fitting both Z1 and Z2.
          A value greater than 0.5 puts more weight on EIS,
          while a value less than 0.5 puts more weight on 2nd-NLEIS.
          (i.e. cost*cost_func(Z1)+(1-cost)*cost_func(Z2))

        - max_f : float, optional

          The maximum frequency of interest for 2nd-NLEIS. Default is np.inf.
          Available only with EISandNLEIS and NLESCustomCircuit circuit.

    Returns
    -------
    results : dict
        A dictionary containing the fitting results:

        - 'idx': int or None, the index of the run if provided.

        - 'Status': bool, whether the fit was successful.

        - 'p0': array_like, the initial guess used for the fit.

        - 'p': array_like, the fitted parameters if the fit was successful.

        - 'cost': float, the cost value of the fit if successful.

        - 'model': the fitted model instance if the fit was successful.

        - 'err': str, the error message if the fit failed.
    """

    try:
        # Validate inputs
        Z1_data = fit_kwargs.pop("Z1", None)
        Z2_data = fit_kwargs.pop("Z2", None)
        impedance_data = fit_kwargs.pop("impedance", None)

        # Make a private copy (works for both processes & threads)
        model = base_model

        # Set the per-try initial guess
        model.initial_guess = list(initial_guess)
        cost_func = fit_kwargs.pop("cost_func", cost_max_norm)
        cost = fit_kwargs.pop("cost", 0.5)

        if (Z1_data is not None) and (Z2_data is not None) and (impedance_data
                                                                is None):
            # Avoid BLAS/OpenMP oversubscription inside each worker
            with threadpool_limits(1):
                model.fit(f, Z1_data, Z2_data, **fit_kwargs)
            Z1_fit, Z2_fit = model.predict(f)

            cost_value = cost_func(Z1_data, Z1_fit) + \
                cost * cost_func(Z2_data, Z2_fit)
        elif (impedance_data is not None) and (Z1_data is None) and (Z2_data
                                                                     is None):
            # Avoid BLAS/OpenMP oversubscription inside each worker

            with threadpool_limits(1):
                model.fit(f, impedance_data, **fit_kwargs)
            Z_fit = model.predict(f)
            cost_value = cost_func(impedance_data, Z_fit)
        elif (Z1_data is not None or Z2_data is not None) and (impedance_data
                                                               is not None):
            raise ValueError(
                "Provide either (Z1 and Z2) or impedance, not both.")

        else:
            raise ValueError(
                "Either (Z1 and Z2) or "
                "impedance must be provided for fitting.")

        results = {'idx': run_idx, 'Status': True, 'p0': initial_guess,
                   'p': model.parameters_, 'cost': cost_value, 'model': model}
        if show_results:
            print(results)
        return results
    except Exception as e:
        # add traceback to the results for easier debugging
        results = {'idx': run_idx, 'Status': False, 'p0': initial_guess,
                   'err': repr(e), 'traceback': traceback.format_exc(),
                   }
        if show_results:
            print(results)
        return results


def multistart_fit(base_model, f, sampling_method="sobol", num_samples=1024,
                   n_jobs=-1, backend="loky", batch_size="auto", **fit_kwargs):
    """
    Try many initial guesses in parallel.
    This can help to avoid local minima in the optimization.

    Parameters
    ----------
    base_model : instance of circuit for EIS, NLEIS, or EISandNLEIS
        The model to be fitted. The initial_guess attribute will be used as
        the starting point for the optimization.
    f : array_like
        Frequencies at which the data is measured.
    sampling_method : str, optional. The default is 'sobol'.
        Method to sample the initial guesses. Supported methods are 'sobol',
        'random', or 'custom'.
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
        from CustomCircuit, NLESCustomCircuit, or EISandNLEIS.
        Supported keywords are:

        - Z1: array_like, the EIS data to fit.

          For simultaneous EIS and 2nd-NLEIS fitting.

        - Z2: array_like, the 2nd-NLEIS data to fit.

          For simultaneous EIS and 2nd-NLEIS fitting.

        - impedance: array_like, the impedance data to fit.

          For EIS or NLEIS only fitting.

        - cost_func: callable, optional. The default is cost_max_norm.

          It takes two arguments:
          the measured data and the fitted data,
          and returns a scalar cost value.
          Other cost functions will be supported in the future.

        - cost: float, The default is 0.5.

          The weight of the cost function when fitting both Z1 and Z2.
          A value greater than 0.5 puts more weight on EIS,
          while a value less than 0.5 puts more weight on 2nd-NLEIS.
          (i.e. cost*cost_func(Z1)+(1-cost)*cost_func(Z2))

        - max_f : float, optional. Default is np.inf

          The maximum frequency of interest for 2nd-NLEIS

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
            'bounds': [[min(guess, 0), max(2*guess, 0)]
                       for guess in initial_guess]
        }
        initial_guesses = sobol.sample(problem, N=num_samples)
    elif sampling_method == "random":
        initial_guesses = np.random.uniform(low=0, high=2, size=(
            num_samples, len(initial_guess)))*initial_guess
    elif sampling_method == 'custom':
        if 'initial_guesses' not in fit_kwargs:
            raise ValueError(
                "For 'custom' sampling_method, "
                "'initial_guesses' must be provided in fit_kwargs.")
        initial_guesses = fit_kwargs.pop('initial_guesses')

    else:
        raise ValueError(
            "Unsupported sampling method, only 'sobol', 'random' "
            "and 'custom' are supported currently.")
    # Adding the sampled initial guess to the inputed initial guess
    initial_guesses = np.vstack([initial_guess, initial_guesses])

    # Create and launch the jobs
    jobs = (
        delayed(fit_once)(base_model, p0, f, run_idx=i, **fit_kwargs)
        for i, p0 in enumerate(initial_guesses)
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
    print("Best fitting result: \nidx:", best['idx'],
          "\np0:", best["p0"],
          "\np:", best["p"],
          "\ncost:", best["cost"])
    return best, results


def batch_data_fit(base_model, f, impedance_list=None, Z1_list=None,
                   Z2_list=None,
                   n_jobs=-1, backend="loky", batch_size="auto", **fit_kwargs):
    """

    Perform parallel fitting using the same circuit model to multiple data
    using the same initial guesses.
    This can be a useful approach to do batch analysis on a big dataset.

    Parameters
    ----------
    base_model : object
        The base circuit model.
    f : frequencies
        Frequencies at which the data is measured.
    impedance_list : list of array_like, optional
        List of impedance data to fit. For EIS or NLEIS only fitting.
    Z1_list : list of array_like, optional
        List of EIS data to fit. For simultaneous EIS and 2nd-NLEIS fitting.
    Z2_list : list of array_like, optional
        List of 2nd-NLEIS data to fit. For simultaneous EIS and 2nd-NLEIS
        fitting.
    n_jobs : int, optional
        The number of jobs to run in parallel. The default is -1, which means
        using all available processors.
    backend : str, optional
        The parallelization backend to use. The default is 'loky'.
    batch_size : int or str, optional
        The number of tasks to dispatch at once to each worker. The default is
        'auto'.
    **fit_kwargs : keyword arguments
        Additional keyword arguments to pass to the fit method of the model.
        These models can be from CustomCircuit,
        NLESCustomCircuit, or EISandNLEIS.

        Supported keywords are:

        - cost_func: callable, optional. The default is cost_max_norm.

          It takes two arguments:
          the measured data and the fitted data,
          and returns a scalar cost value.
          Other cost functions will be supported in the future.

        - cost: float, optional. The default is 0.5.

          The weight of the cost function when fitting both Z1 and Z2.
          A value greater than 0.5 puts more weight on EIS,
          while a value less than 0.5 puts more weight on 2nd-NLEIS.
          (i.e. cost*cost_func(Z1)+(1-cost)*cost_func(Z2))

        - max_f : float, optional. The default is np.inf.

            The maximum frequency of interest for 2nd-NLEIS.

    Returns
    -------
    results : list of dict
        All fitting results.
    """
    initial_guess = base_model.initial_guess

    # Create and launch the jobs: loop over datasets (Z1 and Z2 or impedance)
    # to fit

    if (((Z1_list is not None) and (Z2_list is not None))
            and (impedance_list is None)):
        jobs = (
            delayed(fit_once)(base_model, initial_guess, f, run_idx=i,
                              Z1=Z1_data, Z2=Z2_data, **fit_kwargs)
            for i, (Z1_data, Z2_data) in enumerate(zip(Z1_list, Z2_list))
        )

    elif ((impedance_list is not None) and ((Z1_list is None)
          and (Z2_list is None))):
        jobs = (
            delayed(fit_once)(base_model, initial_guess, f, run_idx=i,
                              impedance=impedance, **fit_kwargs)
            for i, impedance in enumerate(impedance_list)
        )
    elif (((Z1_list is not None) or (Z2_list is not None))
          and (impedance_list is not None)):
        raise ValueError(
            "Provide either (Z1_list and Z2_list) or "
            "impedance_list, not both.")
    else:
        raise ValueError(
            "Either (Z1_list and Z2_list) or impedance_list must be provided"
            " for fitting.")
    # Run the jobs in parallel
    results = Parallel(n_jobs=n_jobs, backend=backend,
                       batch_size=batch_size)(jobs)

    # Filter out failed fits
    fits = [r for r in results if r.get("Status")]
    if not fits:
        raise RuntimeError("All fits failed", results)

    return results


def batch_model_fit(base_models, f,
                    n_jobs=-1, backend="loky", batch_size="auto",
                    **fit_kwargs):
    """

    Perform parallel fitting of the same data with multiple models.
    This can be an useful approach to do model selection.

    Parameters
    ----------

    base_models : list of objects
        The base circuit models.
    f : array_like
        Frequencies at which the data is measured.
    n_jobs : int, optional
        The number of jobs to run in parallel. The default is -1, which means
        using all available processors.
    backend : str, optional
        The parallelization backend to use. The default is 'loky'.
    batch_size : int or str, optional
        The number of tasks to dispatch at once to each worker. The default is
        'auto'.
    **fit_kwargs : keyword arguments
        Additional keyword arguments to pass to the fit method of the model.
        These models can be from CustomCircuit,
        NLESCustomCircuit, or EISandNLEIS.

        Supported keywords are:

        - Z1: array_like, the EIS data to fit.

          For simultaneous EIS and 2nd-NLEIS fitting.

        - Z2: array_like, the 2nd-NLEIS data to fit.

          For simultaneous EIS and 2nd-NLEIS fitting.

        - impedance: array_like, the impedance data to fit.

          For EIS or NLEIS only fitting.

        - cost_func: callable, optional. The default is cost_max_norm.

          It takes two arguments:
          the measured data and the fitted data,
          and returns a scalar cost value.
          Other cost functions will be supported in the future.

        - cost: float, optional. The default is 0.5.

          The weight of the cost function when fitting both Z1 and Z2.
          A value greater than 0.5 puts more weight on EIS,
          while a value less than 0.5 puts more weight on 2nd-NLEIS.
          (i.e. cost*cost_func(Z1)+(1-cost)*cost_func(Z2))

        - max_f : float, optional. The default is np.inf.

          The maximum frequency of interest for 2nd-NLEIS.

    Returns
    -------
    results : list of dict
        All fitting results.
    """
    # Create and launch the jobs: loop over models to fit
    jobs = (
        delayed(fit_once)(base_model, base_model.initial_guess, f, run_idx=i,
                          **fit_kwargs)
        for i, base_model in enumerate(base_models)
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
    print("Best fitting result: \nidx:", best['idx'],
          "\np0:", best["p0"],
          "\np:", best["p"],
          "\ncost:", best["cost"])
    return best, results
