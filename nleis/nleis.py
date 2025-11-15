from .nleis_fitting import simul_fit, \
    wrappedImpedance, individual_parameters
from .nleis_elements_pair import *  # noqa: F401, F403
from impedance.models.circuits.fitting import check_and_eval
from impedance.models.circuits.elements import circuit_elements, \
    get_element_from_name
from impedance.models.circuits.circuits import BaseCircuit
from .fitting import circuit_fit, buildCircuit, calculateCircuitLength, \
    extract_circuit_elements

from impedance.visualization import plot_bode
from .visualization import plot_altair, plot_first, plot_second
from nleis.fitting import CircuitGraph

import json
import matplotlib.pyplot as plt
import numpy as np
import warnings


class EISandNLEIS:
    # ToDO: add SSO method and SO method
    def __init__(self, circuit_1='', circuit_2='', initial_guess=[],
                 constants=None, name=None, graph=False, **kwargs):
        """
            Constructor for a customizable linear and nonlinear
            equivalent circuit model
            to enable simultaneous EIS and 2nd-NLEIS analysis.

            Parameters
            ----------
            circuit_1 : str
                A string representing the equivalent circuit for linear EIS.

            circuit_2 : str
                A string representing the equivalent circuit for 2nd-NLEIS.

            initial_guess : numpy.ndarray
                Initial guess values for the circuit parameters.

            constants : dict, optional
                Dictionary of parameters and values
                to hold constant during fitting (e.g., {"R0": 0.1}).
                These constants can apply to both EIS and 2nd-NLEIS elements.
                Note:
                The initial guess for constant parameters should be excluded.

            name : str, optional
                A name for the model.
            graph : bool, optional
                Whether to use execution graph to process the circuit.
                Defaults to False, which uses eval based code

            Notes
            -----
            The custom circuit for EIS is defined as a string with elements
            in series (separated by `-`) and
            elements in parallel (grouped as p(x, y)).

            For 2nd-NLEIS, the circuit should be grouped by `d(cathode, anode)`
            to denote difference calculations.

            Each element can be appended with an integer (e.g., R0)
            or an underscore and an integer (e.g., CPE_1)
            to help track multiple elements of the same type.

            Example
            -------
            A two-electrode cell with a spherical porous cathode and anode,
            resistor, and inductor can be represented as:

            EIS: circuit_1 = 'L0-R0-TDS0-TDS1'
            NLEIS: circuit_2 = 'd(TDSn0-TDSn1)'

        """
        self.graph = graph
        # if supplied, check that initial_guess is valid and store
        initial_guess = [x for x in initial_guess if x is not None]
        for i in initial_guess:
            if not isinstance(i, (float, int, np.int32, np.float64)):
                raise TypeError(f'value {i} in initial_guess is not a number')

        # initalize class attributes
        # self.initial_guess = list(initial_guess)
        self.initial_guess = initial_guess

        self.circuit_1 = circuit_1
        self.circuit_2 = circuit_2
        elements_1 = extract_circuit_elements(circuit_1)
        elements_2 = extract_circuit_elements(circuit_2)

        # new code for circuit length calculation using circuit element
        # producing edited circuit
        # i.e. circuit_1 = L0-R0-TDP0-TDS1;
        # circuit_2 = TDPn0-TDSn1; edited_circuit = L0-R0-TDPn0-TDSn1
        if elements_1 != [''] or elements_2 != ['']:
            if elements_1 == [''] or elements_2 == ['']:
                raise ValueError(
                    'Either circuit_1 or circuit_2 cannot be empty')
            edited_circuit = ''
            for elem in elements_1:
                raw_elem = get_element_from_name(elem)
                len_raw_elem = len(raw_elem)
                nl_elem = elem[0:len_raw_elem] + 'n' + elem[len_raw_elem:]
                # elem[0:-1]+'n'+elem[-1]
                if nl_elem in elements_2:
                    edited_circuit += '-' + nl_elem
                else:
                    edited_circuit += '-' + elem
            self.edited_circuit = edited_circuit[1:]
        else:
            self.edited_circuit = ''

        circuit_len = calculateCircuitLength(self.edited_circuit)

        for element in elements_2:
            if ((element.replace('n', '') not in elements_1) or
                    ('n' not in element and element != '')):
                raise TypeError(
                    'The pairing of linear and nonlinear elements must be'
                    + 'presented correctly for simultaneous fitting.'
                    + ' Double check the EIS circuit: '
                    + f'{circuit_1}' + ' and the NLEIS circuit: '
                    + f'{circuit_2}' + '. The parsed elements are '
                    + f'{elements_1}'+' for circuit_1, and '
                    + f'{elements_2}' + ' for circuit_2.')

        input_elements = elements_1 + elements_2

        if constants is not None:
            self.constants = constants
            self.constants_1 = {}
            self.constants_2 = {}
            # New code for constant separatation
            # in the current development the differentiation
            # between linear and nonlinear is separated by 'n'
            # using get_element_from_name to get the raw element
            # and adding n to the end

            # The constant for EIS circuit will present in
            # both constants_1 and constants_2
            # The constants for 2nd-NLEIS will only present in
            # constants_2
            for elem in self.constants:

                raw_elem = get_element_from_name(elem)
                raw_circuit_elem = elem.split('_')[0]

                if raw_circuit_elem not in input_elements:
                    raise ValueError(f'{raw_elem} not in ' +
                                     f'input elements ({input_elements})')
                raw_num_params = check_and_eval(raw_elem).num_params
                if raw_num_params <= 1:
                    # currently there is no single parameter nonlinear element,
                    # so this can work, but might be changed in the future
                    self.constants_1[elem] = self.constants[elem]
                else:
                    param_num = int(elem.split('_')[-1])
                    if raw_elem[-1] != 'n':

                        if param_num >= raw_num_params:
                            raise ValueError(
                                f'{elem} is out of the range of'
                                + ' the maximum allowed '
                                + f'number of parameters ({raw_num_params})')

                        self.constants_1[elem] = self.constants[elem]
                        len_elem = len(raw_elem)
                        nl_elem = elem[0:len_elem]+'n'+elem[len_elem:]
                        raw_nl_elem = get_element_from_name(nl_elem)
                        allowed_elems = circuit_elements.keys()
                        if raw_nl_elem in allowed_elems:
                            self.constants_2[nl_elem] = self.constants[elem]
                        else:
                            # this code is kept here to ignore
                            # EIS only constants in constnats_2
                            self.constants_2[elem] = self.constants[elem]

                    if raw_elem[-1] == 'n':

                        if param_num >= raw_num_params:
                            raise ValueError(
                                f'{elem} is out of the range of'
                                + ' the maximum allowed '
                                + f'number of parameters ({raw_num_params})')

                        num_params = check_and_eval(raw_elem[0:-1]).num_params
                        if param_num < num_params:
                            self.constants_1[elem.replace(
                                'n', '')] = self.constants[elem]
                        self.constants_2[elem] = self.constants[elem]

        else:
            self.constants = {}
            self.constants_1 = {}
            self.constants_2 = {}

        if len(self.initial_guess) + len(self.constants) != circuit_len:
            raise ValueError('The number of initial guesses ' +
                             f'({len(self.initial_guess)}) + ' +
                             'the number of constants ' +
                             f'({len(self.constants)})' +
                             ' must be equal to ' +
                             f'the circuit length ({circuit_len})')
        self.name = name

        # initialize fit parameters and confidence intervals
        self.parameters_ = None
        self.conf_ = None

        self.p1, self.p2 = individual_parameters(
            self.edited_circuit, self.initial_guess,
            self.constants_1, self.constants_2)
        if self.circuit_1:
            self.cg1 = CircuitGraph(self.circuit_1, self.constants_1)
        if self.circuit_2:
            self.cg2 = CircuitGraph(self.circuit_2, self.constants_2)

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            matches = []
            for key, value in self.__dict__.items():
                if isinstance(value, np.ndarray):
                    matches.append((value == other.__dict__[key]).all())
                else:
                    matches.append(value == other.__dict__[key])
            return np.array(matches).all()
        else:
            raise TypeError('Comparing object is not of the same type.')

    def fit(self, frequencies, Z1, Z2, bounds=None,
            opt='max', max_f=np.inf, cost=0.5, param_norm=True,
            positive=True, **kwargs):
        """

        Fit the EIS and 2nd-NLEIS circuit model simultaneously.

        Parameters
        ----------
        frequencies : numpy.ndarray
            Array of frequencies.

        Z1 : numpy.ndarray, dtype=complex128
            EIS values to fit.

        Z2 : numpy.ndarray, dtype=complex128
            2nd-NLEIS values to fit.

        bounds : tuple of array_like, optional
            A 2-tuple representing the lower and upper bounds on parameters.
            If bounds are provided, the input will be normalized to stabilize
            the algorithm.

        opt : str, optional
            Optimization method to use. Default is 'max'. 'max' refers to the
            maximum normalization method introduced by Ji and Schwartz. [1]

            The 'neg' option is available to perform negative log-likelihood
            calculation, as introduced by Kirk et al. [2]

        max_f : float, optional
            The maximum frequency of interest for 2nd-NLEIS. Default is np.inf.

        cost : float, optional
            Cost function weight for optimization. Default is 0.5.
            Overall cost = cost * EIS + (1 - cost) * 2nd-NLEIS.

        param_norm : bool, optional
            Whether to apply parameter normalization. Default is True.

        positive : bool, optional
            Whether to restrict the EIS values to the positive real quadrant.
            Default is True.

        kwargs :
            Additional keyword arguments passed to `simul_fit`,
            and subsequently to `scipy.optimize.curve_fit`.

        Returns
        -------
        self : object
            Returns the instance of the model for chaining.

        References
        ----------
        [1] Y. Ji, D.T. Schwartz,
        Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
        II.Model-Based Analysis of Lithium-Ion Battery Experiments.
        J. Electrochem. Soc., 2024. `doi:10.1149/1945-7111/ad2596
        <https://iopscience.iop.org/article/10.1149/1945-7111/ad2596>`_.

        [2] Kirk et al.,
        Nonlinear Electrochemical Impedance Spectroscopy for Lithium-Ion
        Battery Model Parameterization.
        J. Electrochem. Soc., 2023. `doi:10.1149/1945-7111/acada7
        <https://iopscience.iop.org/article/10.1149/1945-7111/acada7>`_.

        """
        # check that inputs are valid:
        #    frequencies: array of numbers
        #    impedance: array of complex numbers
        #    impedance and frequency match in length

        frequencies = np.array(frequencies, dtype=float)
        Z1 = np.array(Z1, dtype=complex)
        Z2 = np.array(Z2, dtype=complex)
        if len(frequencies) != len(Z1):
            raise ValueError(
                'length of frequencies and impedance do not match for EIS')

        if self.initial_guess != []:
            parameters, conf = simul_fit(
                frequencies, Z1, Z2, self.circuit_1, self.circuit_2,
                self.edited_circuit, self.initial_guess,
                constants_1=self.constants_1,
                constants_2=self.constants_2, bounds=bounds, opt=opt,
                cost=cost, max_f=max_f, param_norm=param_norm,
                positive=positive, graph=self.graph,
                **kwargs)
            # self.parameters_ = list(parameters)
            self.parameters_ = parameters

            if conf is not None:
                # self.conf_ = list(conf)
                self.conf_ = conf
                self.conf1, self.conf2 = individual_parameters(
                    self.edited_circuit, self.conf_, self.constants_1,
                    self.constants_2)
            # if cov is not None:
            #     self.cov_ = cov
            #     self.cov1, self.cov2 = individual_parameters(
            #         self.edited_circuit, self.cov_, self.constants_1,
            #         self.constants_2)

            self.p1, self.p2 = individual_parameters(
                self.edited_circuit, self.parameters_, self.constants_1,
                self.constants_2)
        else:
            raise ValueError('no initial guess supplied')

        return self

    def _is_fit(self):
        """

        check if model has been fit (parameters_ is not None)

        """
        if self.parameters_ is not None:
            return True
        else:
            return False

    def predict(self, frequencies, max_f=np.inf, use_initial=False):
        """
        Predict EIS and 2nd-NLEIS using
        linear and nonlinear equivalent circuit model.

        Parameters
        ----------
        frequencies : numpy.ndarray
            Array of frequency values.

        max_f : float, optional
            The maximum frequency of interest for 2nd-NLEIS. Default is np.inf.

        use_initial : bool
            If True and the model was previously fit,
            use the initial parameters instead of the fitted ones.

        Returns
        -------
        x1 : numpy.ndarray
            Calculated impedance for EIS (Z1).

        x2 : numpy.ndarray
            Calculated impedance for 2nd-NLEIS (Z2).

        """

        if not isinstance(frequencies, np.ndarray):
            raise TypeError('frequencies is not of type np.ndarray')
        if not (np.issubdtype(frequencies.dtype, np.integer) or
                np.issubdtype(frequencies.dtype, np.floating)):
            raise TypeError('frequencies array should have a numeric ' +
                            f'dtype (currently {frequencies.dtype})')

        f1 = frequencies
        mask = np.array(frequencies) < max_f
        f2 = frequencies[mask]

        if self._is_fit() and not use_initial:
            x1, x2 = wrappedImpedance(self.edited_circuit,
                                      self.circuit_1, self.constants_1,
                                      self.circuit_2, self.constants_2, f1, f2,
                                      self.parameters_, graph=self.graph)
            return x1, x2
        else:
            warnings.warn("Simulating circuit based on initial parameters")
            x1, x2 = wrappedImpedance(self.edited_circuit,
                                      self.circuit_1, self.constants_1,
                                      self.circuit_2, self.constants_2, f1, f2,
                                      self.initial_guess, graph=self.graph)
            return x1, x2

    def get_param_names(self, circuit, constants):
        """
        Converts a circuit string into
        a list of parameter names and their units.

        Parameters
        ----------
        circuit : str
            A string representing the circuit. This string may contain elements
            connected in series or parallel, and may also include
            nonlinear terms represented by 'd' for NLEIS elements.

        constants : dict
            A dictionary of constant parameters with their values.
            Parameters that are constants are excluded from the output.

        Returns
        -------
        full_names : list of str
            A list containing the full names of the parameters derived from the
            circuit string.

        all_units : list of str
            A list containing the corresponding units of the parameters.
        """

        # parse the element names from the circuit string
        names = circuit.replace('d', '').replace(
            '(', '').replace(')', '')  # edit for nleis.py

        names = names.replace('p', '').replace('(', '').replace(')', '')
        names = names.replace(',', '-').replace(' ', '').split('-')

        full_names, all_units = [], []
        for name in names:
            elem = get_element_from_name(name)
            num_params = check_and_eval(elem).num_params
            units = check_and_eval(elem).units
            if num_params > 1:
                for j in range(num_params):
                    full_name = '{}_{}'.format(name, j)
                    if full_name not in constants.keys():
                        full_names.append(full_name)
                        all_units.append(units[j])
            else:
                if name not in constants.keys():
                    full_names.append(name)
                    all_units.append(units[0])

        return full_names, all_units

    def __str__(self):
        """

        Defines the pretty printing of the circuit
        for both EIS and 2nd-NLEIS

        """

        to_print = '\n'
        if self.name is not None:
            to_print += 'Name: {}\n'.format(self.name)
        to_print += 'EIS Circuit string: {}\n'.format(self.circuit_1)
        to_print += 'NLEIS Circuit string: {}\n'.format(self.circuit_2)
        to_print += "Fit: {}\n".format(self._is_fit())

        if len(self.constants) > 0:
            to_print += '\nConstants:\n'
            for name, value in self.constants.items():
                elem = get_element_from_name(name)
                units = check_and_eval(elem).units
                if '_' in name:
                    unit = units[int(name.split('_')[-1])]
                else:
                    unit = units[0]
                to_print += '  {:>5} = {:.2e} [{}]\n'.format(name, value, unit)

        names1, units1 = self.get_param_names(self.circuit_1, self.constants_1)
        names2, units2 = self.get_param_names(self.circuit_2, self.constants_2)

        to_print += '\nEIS Initial guesses:\n'
        p1, p2 = individual_parameters(
            self.edited_circuit, self.initial_guess, self.constants_1,
            self.constants_2)
        for name, unit, param in zip(names1, units1, p1):
            to_print += '  {:>5} = {:.2e} [{}]\n'.format(name, param, unit)
        to_print += '\nNLEIS Initial guesses:\n'
        for name, unit, param in zip(names2, units2, p2):
            to_print += '  {:>5} = {:.2e} [{}]\n'.format(name, param, unit)
        if self._is_fit():
            params1, confs1 = self.p1, self.conf1
            to_print += '\nEIS Fit parameters:\n'
            for name, unit, param, conf in zip(names1, units1,
                                               params1, confs1):
                to_print += '  {:>5} = {:.2e}'.format(name, param)
                to_print += '  (+/- {:.2e}) [{}]\n'.format(conf, unit)
            params2, confs2 = self.p2, self.conf2
            to_print += '\nNLEIS Fit parameters:\n'
            for name, unit, param, conf in zip(names2, units2,
                                               params2, confs2):
                to_print += '  {:>5} = {:.2e}'.format(name, param)
                to_print += '  (+/- {:.2e}) [{}]\n'.format(conf, unit)

        return to_print

    def extract(self):
        """
        Extracts the parameters of
        EIS and 2nd-NLEIS circuits into dictionaries.

        This method retrieves the parameter names from
        `circuit_1` (EIS circuit) and `circuit_2` (NLEIS circuit),
        and maps the fitted parameters to their respective names.

        Returns
        -------
        dict1 : dict
            A dictionary containing the parameters and
            their values for `circuit_1` (EIS circuit).

        dict2 : dict
            A dictionary containing the parameters and
            their values for `circuit_2` (NLEIS circuit).
        """

        names1, units1 = self.get_param_names(self.circuit_1, self.constants_1)
        dict1 = {}
        if self._is_fit():
            params1 = self.p1
            for names, param in zip(names1, params1):
                dict1[names] = param

        names2, units2 = self.get_param_names(self.circuit_2, self.constants_2)
        dict2 = {}
        if self._is_fit():
            params2 = self.p2

            for names, param in zip(names2, params2):
                dict2[names] = param

        return dict1, dict2

    def plot(self, ax=None, f_data=None, Z1_data=None, Z2_data=None,
             kind='nyquist', max_f=np.inf, **kwargs):
        """
        Visualizes the model and optional data as Nyquist, Bode,
        or Altair (interactive) plots.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes will be created.

        f_data : numpy.ndarray, optional
            Array of frequency values for input data (used for Bode plots).
            The default is None.

        Z1_data : numpy.ndarray of complex, optional
            Array of EIS values (impedance data). The default is None.

        Z2_data : numpy.ndarray of complex, optional
            Array of 2nd-NLEIS values (impedance data). The default is None.

        kind : {'altair', 'nyquist', 'bode'}, optional
            The type of plot to visualize.

            - 'nyquist': Nyquist plot of real vs imaginary impedance.

            - 'bode': Bode plot showing magnitude and phase.

            - 'altair': Altair plot for interactive visualizations.

            Default is 'nyquist'.

        max_f : float, optional
            The maximum frequency of interest for 2nd-NLEIS data.
            The default is np.inf.

        **kwargs : optional
            Additional keyword arguments passed to `matplotlib.pyplot.Line2D`
            (for 'nyquist' or 'bode') to specify properties like linewidth,
            color, marker type, etc.
            If `kind` is 'altair',
            `kwargs` is used to specify plot height as `size`.

        Returns
        -------
        ax : matplotlib.axes.Axes or tuple of Axes, optional
            Axes object(s) with the plotted data
            if 'nyquist' or 'bode' plot is used.

        chart1, chart2 : altair.Chart, optional
            If `kind` is 'altair', it returns two Altair chart objects,
            one for EIS data and one for 2nd-NLEIS data.

        Raises
        ------
        ValueError
            If an unsupported `kind` is provided.
        """

        if kind == 'nyquist':
            if ax is None:
                _, ax = plt.subplots(1, 2, figsize=(10, 5))

            # we don't need the if else statement if we want
            # to enable plot without fit
            # if self._is_fit():
            if f_data is not None:
                f_pred = f_data
            else:
                f_pred = np.logspace(5, -3)

            if Z1_data is not None:
                ax[0] = plot_first(ax[0], Z1_data, scale=1, fmt='s', **kwargs)
                # impedance.py style
                # plot_nyquist(Z1_data, ls='', marker='s', ax=ax[0], **kwargs)
            if Z2_data is not None:
                if f_data is not None:
                    mask = np.array(f_data) < max_f
                    ax[1] = plot_second(ax[1], Z2_data[mask],
                                        scale=1, fmt='s', **kwargs)
                else:
                    ax[1] = plot_second(ax[1], Z2_data,
                                        scale=1, fmt='s', **kwargs)
                # impedance.py style
                # plot_nyquist(Z2_data, units='Ohms/A', ls='',
                # marker='s', ax=ax[1], **kwargs)

            Z1_fit, Z2_fit = self.predict(f_pred, max_f=max_f)
            ax[0] = plot_first(ax[0], Z1_fit, scale=1, fmt='-', **kwargs)
            # plot_nyquist(Z1_fit, ls='-', marker='', ax=ax[0], **kwargs)
            ax[1] = plot_second(ax[1], Z2_fit, scale=1, fmt='-', **kwargs)
            # plot_nyquist(Z2_fit,units='Ohms/A', ls='-',
            # marker='', ax=ax[1], **kwargs)

            ax[0].legend(['Data', 'Fit'])
            ax[1].legend(['Data', 'Fit'])
            return ax
        elif kind == 'bode':
            if ax is None:
                _, ax = plt.subplots(2, 2, figsize=(8, 8))

            if f_data is not None:
                f_pred = f_data
            else:
                f_pred = np.logspace(5, -3)

            if Z1_data is not None:
                if f_data is None:
                    raise ValueError('f_data must be specified if' +
                                     ' Z_data for a Bode plot')
                ax[:, 0] = plot_bode(f_data, Z1_data, ls='', marker='s',
                                     axes=ax[:, 0], **kwargs)
            if Z2_data is not None:
                if f_data is None:
                    raise ValueError('f_data must be specified if' +
                                     ' Z_data for a Bode plot')
                mask = np.array(f_pred) < max_f
                f2 = f_data[mask]
                Z2 = Z2_data[mask]
                ax[:, 1] = plot_bode(f2, Z2, units='Ω/A', ls='', marker='s',
                                     axes=ax[:, 1], **kwargs)
            # we don't need the if else statement
            # if we want to enable plot without fit
            # if self._is_fit():
            Z1_fit, Z2_fit = self.predict(f_pred, max_f=max_f)

            f1 = f_pred
            f2 = f_pred[np.array(f_pred) < max_f]

            ax[:, 0] = plot_bode(f1, Z1_fit, ls='-', marker='o',
                                 axes=ax[:, 0], **kwargs)
            ax[:, 1] = plot_bode(f2, Z2_fit, units='Ω/A', ls='-',
                                 marker='o',
                                 axes=ax[:, 1], **kwargs)

            ax[0, 0].set_ylabel(r'$|Z_{1}(\omega)|$ ' +
                                '$[{}]$'.format('Ω'), fontsize=20)
            ax[1, 0].set_ylabel(
                r'$-\phi_{Z_{1}}(\omega)$ ' + r'$[^o]$', fontsize=20)
            ax[0, 1].set_ylabel(r'$|Z_{2}(\omega)|$ ' +
                                '$[{}]$'.format('Ω/A'), fontsize=20)
            ax[1, 1].set_ylabel(
                r'$-\phi_{Z_{2}}(\omega)$ ' + r'$[^o]$', fontsize=20)
            ax[0, 0].legend(['Data', 'Fit'], fontsize=20)
            ax[0, 1].legend(['Data', 'Fit'], fontsize=20)
            ax[1, 0].legend(['Data', 'Fit'], fontsize=20)
            ax[1, 1].legend(['Data', 'Fit'], fontsize=20)
            return ax
        elif kind == 'altair':
            plot_dict_1 = {}
            plot_dict_2 = {}

            if ((Z1_data is not None) and (Z2_data is not None) and (
                    f_data is not None)):
                plot_dict_1['data'] = {'f': f_data, 'Z': Z1_data}
                mask = np.array(f_data) < max_f
                plot_dict_2['data'] = {'f': f_data[mask], 'Z': Z2_data[mask]}
            # we don't need the if else statement
            # if we want to enable plot without fit
            # if self._is_fit():
            if f_data is not None:

                f_pred = f_data

            else:

                f_pred = np.logspace(5, -3)

            if self.name is not None:
                name = self.name
            else:
                name = 'fit'

            Z1_fit, Z2_fit = self.predict(f_pred, max_f=max_f)
            mask = np.array(f_pred) < max_f
            plot_dict_1[name] = {'f': f_pred, 'Z': Z1_fit, 'fmt': '-'}
            plot_dict_2[name] = {'f': f_pred[mask], 'Z': Z2_fit, 'fmt': '-'}

            chart1 = plot_altair(plot_dict_1, k=1, units='Ω', **kwargs)
            chart2 = plot_altair(plot_dict_2, k=2, units='Ω/A', **kwargs)

            return chart1, chart2
        else:
            raise ValueError("Kind must be one of 'altair'," +
                             f"'nyquist', or 'bode' (received {kind})")

    def save(self, filepath):
        """
        Exports the model to a JSON file.

        This method saves the current model configuration,
        including circuit strings,
        initial guesses, constants, and (if fitted) the model parameters
        and their confidence intervals, to a JSON file.

        Parameters
        ----------
        filepath : str
            The file path where the model should be saved. The model will be
            serialized to a JSON file at this location.

        """

        model_string_1 = self.circuit_1
        model_string_2 = self.circuit_2
        edited_circuit_str = self.edited_circuit
        model_name = self.name

        initial_guess = self.initial_guess

        if self._is_fit():
            parameters_ = list(self.parameters_)
            model_conf_ = list(self.conf_)
            # model_cov_ = list(self.cov_)
            # parameters_ = self.parameters_
            # model_conf_ = self.conf_
            data_dict = {"Name": model_name,
                         "Circuit String 1": model_string_1,
                         "Circuit String 2": model_string_2,
                         "Initial Guess": initial_guess,
                         "Constants": self.constants,
                         "Constants 1": self.constants_1,
                         "Constants 2": self.constants_2,
                         "Fit": True,
                         "Parameters": parameters_,
                         "Confidence": model_conf_,
                         #  "Covariance": model_cov_,
                         "Edited Circuit Str": edited_circuit_str
                         }
        else:
            data_dict = {"Name": model_name,
                         "Circuit String 1": model_string_1,
                         "Circuit String 2": model_string_2,
                         "Initial Guess": initial_guess,
                         "Constants": self.constants,
                         "Constants 1": self.constants_1,
                         "Constants 2": self.constants_2,
                         "Edited Circuit Str": edited_circuit_str,
                         "Fit": False}

        with open(filepath, 'w') as f:
            json.dump(data_dict, f)

    def load(self, filepath, fitted_as_initial=False):
        """
        Imports a model from a JSON file.

        This method loads a saved model from a JSON file
        and restores its circuit strings, initial guesses,
        constants, and (if fitted) parameters. The user
        can choose whether to load the fitted parameters as the initial guess.

        Parameters
        ----------
        filepath : str
            The file path to the JSON file from which to load the model.

        fitted_as_initial : bool, optional
            If True, the model's fitted parameters will be
            loaded as initial guesses.
            Otherwise, the model will load both the initial
            and fitted parameters as a completed model. Default is False.

        """

        json_data_file = open(filepath, 'r')
        json_data = json.load(json_data_file)

        model_name = json_data["Name"]
        model_string_1 = json_data["Circuit String 1"]
        model_string_2 = json_data["Circuit String 2"]
        model_initial_guess = json_data["Initial Guess"]
        model_constants = json_data["Constants"]

        self.initial_guess = model_initial_guess
        self.circuit_1 = model_string_1
        self.circuit_2 = model_string_2
        self.edited_circuit = json_data["Edited Circuit Str"]

        self.constants = model_constants
        self.constants_1 = json_data["Constants 1"]
        self.constants_2 = json_data["Constants 2"]
        self.p1, self.p2 = individual_parameters(
            self.edited_circuit, self.initial_guess,
            self.constants_1, self.constants_2)

        self.cg1 = CircuitGraph(self.circuit_1, self.constants_1)
        self.cg2 = CircuitGraph(self.circuit_2, self.constants_2)

        self.name = model_name

        if json_data["Fit"]:
            if fitted_as_initial:
                self.initial_guess = json_data['Parameters']
                self.p1, self.p2 = individual_parameters(
                    self.edited_circuit, self.initial_guess,
                    self.constants_1, self.constants_2)

            else:
                self.parameters_ = np.array(json_data["Parameters"])
                self.conf_ = np.array(json_data["Confidence"])
                self.conf1, self.conf2 = individual_parameters(
                    self.edited_circuit, self.conf_,
                    self.constants_1, self.constants_2)
                self.p1, self.p2 = individual_parameters(
                    self.edited_circuit, self.parameters_,
                    self.constants_1, self.constants_2)
                # self.cov_ = np.array(json_data["Covariance"])
                # self.cov1, self.cov2 = individual_parameters(
                #     self.edited_circuit, self.cov_,
                #     self.constants_1, self.constants_2)


class NLEISCustomCircuit(BaseCircuit):
    # this class can be fully integrated into CustomCircuit in the future
    # , but for the stable performance of nleis.py, we overwrite it here
    def __init__(self, circuit='', graph=False,  **kwargs):
        """
        Constructor for a customizable nonlinear equivalent circuit model
        for NLEIS analysis.

        Parameters
        ----------
        initial_guess : numpy.ndarray
            Initial guess values for the circuit parameters.

        circuit : str
            A string representing the nonlinear equivalent circuit for NLEIS.

        graph : bool, optional
            Whether to use execution graph to process the circuit.
            Defaults to False, which uses eval based code

        Notes
        -----
        A custom NLEIS circuit is defined as a string comprised of elements
        in series (separated by a `-`),
        elements in parallel (grouped as p(x,y)),
        and elements in difference (grouped as d(x,y)).
        Each element can be appended with an integer (e.g., R0)
        or an underscore and an integer (e.g., CPE_1)
        to help keep track of multiple elements of the same type.

        Example
        -------
        A two-electrode cell with a spherical porous cathode and anode
        is represented as:

            circuit = 'd(TDSn0-TDSn1)'

        """

        super().__init__(**kwargs)
        # self.cov_ = None
        self.circuit = circuit.replace(" ", "")
        self.graph = graph
        if self.circuit:
            self.cg = CircuitGraph(self.circuit, self.constants)

        circuit_len = calculateCircuitLength(self.circuit)

        if len(self.initial_guess) + len(self.constants) != circuit_len:
            raise ValueError('The number of initial guesses ' +
                             f'({len(self.initial_guess)}) + ' +
                             'the number of constants ' +
                             f'({len(self.constants)})' +
                             ' must be equal to ' +
                             f'the circuit length ({circuit_len})')

    def fit(self, frequencies, impedance, bounds=None,
            weight_by_modulus=False, max_f=np.inf, **kwargs):
        """
        Fit the nonlinear equivalent circuit model to NLEIS data.

        Parameters
        ----------
        frequencies : numpy.ndarray
            Array of frequency values.

        impedance : numpy.ndarray, dtype=complex128
            Complex NLEIS impedance values to fit.

        bounds : 2-tuple of array_like, optional
            Lower and upper bounds on the model parameters. If not provided,
            default bounds will be used.

        weight_by_modulus : bool, optional
            If True, uses the modulus of each data point (`|Z|`)
            as the weighting factor.
            This is the standard weighting scheme when experimental variances
            are unavailable. Only applicable when `global_opt=False`.
            Default is False.

        max_f : float, optional
            The maximum frequency of interest for NLEIS.
            This is used for truncating the data based on prior experiments.
            Default is np.inf.

        **kwargs :
            Additional keyword arguments passed to
            `impedance.models.circuits.fitting.circuit_fit`,
            and subsequently to `scipy.optimize.curve_fit`
            or `scipy.optimize.basinhopping`.

        Raises
        ------
        TypeError
            Raised if the length of the frequency data
            does not match the NLEIS impedance data.

        ValueError
            Raised if `initial_guess` is not supplied.

        Returns
        -------
        self : object
            Returns the instance of the model for chaining.
        """
        frequencies = np.array(frequencies, dtype=float)
        impedance = np.array(impedance, dtype=complex)

        if len(frequencies) != len(impedance):
            raise TypeError('length of frequencies and impedance do not match')
        mask = np.array(frequencies) < max_f
        frequencies = frequencies[mask]
        impedance = impedance[mask]

        if self.initial_guess != []:
            parameters, conf = \
                circuit_fit(frequencies, impedance,
                            self.circuit,
                            self.initial_guess,
                            constants=self.constants,
                            bounds=bounds,
                            weight_by_modulus=weight_by_modulus,
                            graph=self.graph,
                            **kwargs)
            self.parameters_ = parameters
            if conf is not None:
                self.conf_ = conf
            # if cov is not None:
            #     self.cov_ = cov
        else:
            raise ValueError('No initial guess supplied')

        return self

    def predict(self, frequencies, max_f=np.inf, use_initial=False):
        """

        Predict impedance using an nonlinear equivalent circuit model

        Parameters
        ----------
        frequencies : numpy.ndarray
            Array of frequency values.

        max_f : float, optional
            The maximum frequency of interest for 2nd-NLEIS. Default is 10.

        use_initial : bool
            If True and the model was previously fit,
            use the initial parameters instead of the fitted ones.

        Returns
        -------
        impedance: ndarray of dtype 'complex128'
            Predicted impedance at each frequency

        """
        frequencies = np.array(frequencies, dtype=float)
        mask = np.array(frequencies) < max_f
        frequencies = frequencies[mask]
        if self.graph:
            self.cg = CircuitGraph(self.circuit, self.constants)

        if self._is_fit() and not use_initial:
            if self.graph:
                return self.cg.compute(frequencies, *self.parameters_)
            else:
                return eval(buildCircuit(self.circuit, frequencies,
                                         *self.parameters_,
                                         constants=self.constants,
                                         eval_string='',
                                         index=0)[0],
                            circuit_elements)
        else:
            warnings.warn("Simulating circuit based on initial parameters")

            if self.graph:
                return self.cg.compute(frequencies, *self.initial_guess)
            else:
                return eval(buildCircuit(self.circuit, frequencies,
                                         *self.initial_guess,
                                         constants=self.constants,
                                         eval_string='',
                                         index=0)[0],
                            circuit_elements)

    def get_param_names(self):
        """

        Converts circuit string to names and units

        """

        # parse the element names from the circuit string
        names = self.circuit.replace('d', '').replace(
            '(', '').replace(')', '')  # edited for nleis.py

        names = names.replace('p', '').replace('(', '').replace(')', '')
        names = names.replace(',', '-').replace(' ', '').split('-')

        full_names, all_units = [], []
        for name in names:
            elem = get_element_from_name(name)
            num_params = check_and_eval(elem).num_params
            units = check_and_eval(elem).units
            if num_params > 1:
                for j in range(num_params):
                    full_name = '{}_{}'.format(name, j)
                    if full_name not in self.constants.keys():
                        full_names.append(full_name)
                        all_units.append(units[j])
            else:
                if name not in self.constants.keys():
                    full_names.append(name)
                    all_units.append(units[0])

        return full_names, all_units

    def extract(self):
        """
        Extracts the parameter names and values of the fitted circuit model.

        This method retrieves the parameter names and units from the circuit
        and maps the fitted parameters to their respective names.
        If the model has been fitted,
        it also includes the fitted values in the output dictionary.

        Returns
        -------
        parameters_dict : dict
            A dictionary where the keys are parameter names,
            and the values are the corresponding fitted parameter values.
        """

        names, units = self.get_param_names()
        dict = {}
        if self._is_fit():
            params, confs = self.parameters_, self.conf_

            for name, unit, param, conf in zip(names, units, params, confs):
                dict[name] = param

        return dict

    def plot(self, ax=None, f_data=None, Z2_data=None,
             kind='nyquist', max_f=np.inf, **kwargs):
        """
        Visualizes the model and optional data as Nyquist, Bode,
        or Altair (interactive) plots.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes will be created.

        f_data : numpy.ndarray, optional
            Array of frequency values for input data (used for Bode plots).
            The default is None.

        Z2_data : numpy.ndarray of complex, optional
            Array of 2nd-NLEIS values (impedance data). The default is None.

        kind : {'altair', 'nyquist', 'bode'}, optional
            The type of plot to visualize.

            - 'nyquist': Nyquist plot of real vs imaginary impedance.

            - 'bode': Bode plot showing magnitude and phase.

            - 'altair': Altair plot for interactive visualizations.

            Default is 'nyquist'.

        max_f : float, optional
            The maximum frequency of interest for 2nd-NLEIS data.
            The default is 10.

        **kwargs : optional
            Additional keyword arguments passed to `matplotlib.pyplot.Line2D`
            (for 'nyquist' or 'bode') to specify properties like linewidth,
            color, marker type, etc.
            If `kind` is 'altair',
            `kwargs` is used to specify plot height as `size`.

        Returns
        -------
        ax : matplotlib.axes.Axes or tuple of Axes, optional
            Axes object(s) with the plotted data
            if 'nyquist' or 'bode' plot is used.

        chart : altair.Chart, optional
            If `kind` is 'altair', it returns Altair chart objects
            for 2nd-NLEIS data.

        Raises
        ------
        ValueError
            If an unsupported `kind` is provided.
        """

        if kind == 'nyquist':
            if ax is None:
                _, ax = plt.subplots(figsize=(5, 5))

            # we don't need the if else statement if we want
            # to enable plot without fit
            # if self._is_fit():
            if f_data is not None:
                f_pred = f_data
            else:
                f_pred = np.logspace(5, -3)

            if Z2_data is not None:
                if f_data is not None:
                    mask = np.array(f_data) < max_f
                    ax = plot_second(ax, Z2_data[mask],
                                     scale=1, fmt='s', **kwargs)
                else:
                    ax = plot_second(ax, Z2_data,
                                     scale=1, fmt='s', **kwargs)
                # impedance.py style
                # plot_nyquist(Z2_data, units='Ohms/A', ls='',
                # marker='s', ax=ax, **kwargs)

            Z2_fit = self.predict(f_pred, max_f=max_f)
            ax = plot_second(ax, Z2_fit, scale=1, fmt='-', **kwargs)
            # plot_nyquist(Z2_fit,units='Ohms/A', ls='-',
            # marker='', ax=ax, **kwargs)

            ax.legend(['Data', 'Fit'])
            return ax
        elif kind == 'bode':
            if ax is None:
                _, ax = plt.subplots(nrows=2, figsize=(8, 8))

            if f_data is not None:
                f_pred = f_data
            else:
                f_pred = np.logspace(5, -3)

            if Z2_data is not None:
                if f_data is None:
                    raise ValueError('f_data must be specified if' +
                                     ' Z_data for a Bode plot')
                mask = np.array(f_pred) < max_f
                f2 = f_data[mask]
                Z2 = Z2_data[mask]
                ax = plot_bode(f2, Z2, units='Ω/A', ls='', marker='s',
                               axes=ax, **kwargs)
            # we don't need the if else statement
            # if we want to enable plot without fit
            # if self._is_fit():
            Z2_fit = self.predict(f_pred, max_f=max_f)

            f2 = f_pred[np.array(f_pred) < max_f]

            ax = plot_bode(f2, Z2_fit, units='Ω/A', ls='-',
                           marker='o',
                           axes=ax, **kwargs)

            ax[0].set_ylabel(r'$|Z_{2}(\omega)|$ ' +
                             '$[{}]$'.format('Ω/A'), fontsize=20)
            ax[1].set_ylabel(
                r'$-\phi_{Z_{2}}(\omega)$ ' + r'$[^o]$', fontsize=20)
            ax[0].legend(['Data', 'Fit'], fontsize=20)
            ax[1].legend(['Data', 'Fit'], fontsize=20)

            return ax
        elif kind == 'altair':
            plot_dict = {}

            if (Z2_data is not None) and (f_data is not None):
                mask = np.array(f_data) < max_f
                plot_dict['data'] = {'f': f_data[mask], 'Z': Z2_data[mask]}
            # we don't need the if else statement
            # if we want to enable plot without fit
            # if self._is_fit():
            if f_data is not None:

                f_pred = f_data

            else:

                f_pred = np.logspace(5, -3)

            if self.name is not None:
                name = self.name
            else:
                name = 'fit'

            Z2_fit = self.predict(f_pred, max_f=max_f)
            mask = np.array(f_pred) < max_f
            plot_dict[name] = {'f': f_pred[mask], 'Z': Z2_fit, 'fmt': '-'}

            chart = plot_altair(plot_dict, k=2, units='Ω/A', **kwargs)

            return chart
        else:
            raise ValueError("Kind must be one of 'altair'," +
                             f"'nyquist', or 'bode' (received {kind})")

    # add on to the load function to create the graph

    def load(self, filepath, fitted_as_initial=False):
        super().load(filepath, fitted_as_initial)
        self.cg = CircuitGraph(self.circuit, self.constants)
