
from .nleis_fitting import simul_fit,wrappedImpedance,individual_parameters
from .nleis_elements_pair import*
from impedance.models.circuits.fitting import check_and_eval
from impedance.models.circuits.elements import circuit_elements,get_element_from_name
from impedance.models.circuits.circuits import BaseCircuit
from .fitting import circuit_fit, buildCircuit, calculateCircuitLength
from .fitting import set_default_bounds,buildCircuit,calculateCircuitLength,extract_circuit_elements
from impedance.models.circuits.fitting import check_and_eval,rmse

from impedance.visualization import plot_bode, plot_nyquist
from .visualization import plot_altair
import json
import matplotlib.pyplot as plt
import numpy as np
import warnings



import json
import matplotlib.pyplot as plt
import numpy as np
import warnings

### ToDO: add SSO method and SO method 
class EISandNLEIS:
    def __init__(self, circuit_1='',circuit_2='',initial_guess = []
                 ,constants = None, name=None, **kwargs):
        """ Constructor for a customizable equivalent circuit model

        Parameters
        ----------

        circuit_1: string
            A string that should be interpreted as an equivalent circuit for linear EIS
            
        circuit_2: string
            A string that should be interpreted as an equivalent circuit for NLEIS
            
        initial_guess: numpy array
                Initial guess of the circuit values
                
        constants : dict, optional
            Parameters and values to hold constant during fitting
            (e.g. {"R0": 0.1})

        name: str, optional
            Name for the model

        Notes
        -----
        A custom circuit is defined as a string comprised of elements in series
        (separated by a `-`) and elements in parallel (grouped as (x,y)) for EIS.
        For NLEIS, the circuit should be grouped by t(cathode,anode)
        Each element can be appended with an integer (e.g. R0) or an underscore
        and an integer (e.g. CPE_1) to make keeping track of multiple elements
        of the same type easier.

        Example:
            A two electrode cell with sperical porous cathode and anode, resistor, and inductor is represents as,
            EIS: circuit_1 = 'L0-R0-TDS0-TDS1'
            NLEIS: circuit_2 = 't(TDSn0-TDSn1)'

        """
        for i in initial_guess:
            if not isinstance(i, (float, int, np.int32, np.float64)):
                raise TypeError(f'value {i} in initial_guess is not a number')

        # initalize class attributes
        self.initial_guess = list(initial_guess)
        self.circuit_1 = circuit_1
        self.circuit_2 = circuit_2
                     
        if constants is not None:
            self.constants = constants
            self.constants_1 = {}
            self.constants_2 = {}
            for name in self.constants:
                if name[3] != 'n':
                    self.constants_1[name]=self.constants[name]
                    self.constants_2[name[0:3]+'n'+name[3:]]=self.constants[name]
                if name[3] == 'n':
                    num_params = check_and_eval(name[0:3]).num_params
                    if int(name[-1])<num_params:
                        self.constants_1[name[0:3]+name[4:]]=self.constants[name]
                    self.constants_2[name]=self.constants[name]
                    

        else:
            self.constants = {}
            self.constants_1 = {}
            self.constants_2 = {}
            
        count = 0
        subtract = 0
        for i in range(len(self.circuit_1)):
            if self.circuit_1[i]=='T' or self.circuit_1[i:i+2]=='RC' :
                count+=1
            if self.circuit_1[i]=='O':
                subtract+=1
            
        circuit_len = calculateCircuitLength(self.circuit_1)+count*2-subtract

        ## to be fixed after finish the naming convention
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




        self.p1, self.p2 = individual_parameters(self.circuit_1,self.initial_guess,self.constants_1,self.constants_2)



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
        
    def fit(self, frequencies, Z1,Z2, bounds = None,
            opt='max',max_f=10,cost = 0.5, **kwargs):
        """ Fit the circuit model

        Parameters
        ----------
        frequencies: numpy array
            Frequencies

        Z1: numpy array of dtype 'complex128'
            EIS values to fit
        Z1: numpy array of dtype 'complex128'
            NLEIS values to fit

        bounds: 2-tuple of array_like, optional
            Lower and upper bounds on parameters. When bounds are provided, the input will normalized to stabilize the algorithm

        Normalization : str, optional
            Default is max normalization. Other normalization will be supported
            in the future 
        max_f: int
            The the maximum frequency of interest for NLEIS

        kwargs :
            Keyword arguments passed to simul_fit,
            and subsequently to scipy.optimize.curve_fit
            or scipy.optimize.basinhopping

        Returns
        -------
        self: returns an instance of self

        """
        # check that inputs are valid:
        #    frequencies: array of numbers
        #    impedance: array of complex numbers
        #    impedance and frequency match in length
        
        frequencies = np.array(frequencies, dtype=float)
        Z1 = np.array(Z1, dtype=complex)
        Z2 = np.array(Z2, dtype=complex)
        if len(frequencies) != len(Z1):
            raise TypeError('length of frequencies and impedance do not match for EIS')

        if self.initial_guess != []:
            parameters, conf = simul_fit(frequencies, Z1,Z2,
                                           self.circuit_1,self.circuit_2, self.initial_guess,
                                           constants=self.constants,
                                           bounds=bounds,
                                           opt=opt,cost = cost,
                                           **kwargs)
            self.parameters_ = list(parameters)
            if conf is not None:
                self.conf_ = list(conf)
                self.conf1, self.conf2 = individual_parameters(self.circuit_1,self.conf_,self.constants_1,self.constants_2)

            self.p1, self.p2 = individual_parameters(self.circuit_1,self.parameters_,self.constants_1,self.constants_2)
        else:
            # TODO auto calculate initial guesses
            raise ValueError('no initial guess supplied')

        return self

    def _is_fit(self):
        """ check if model has been fit (parameters_ is not None) """
        if self.parameters_ is not None:
            return True
        else:
            return False


    def predict(self, frequencies, max_f=10, use_initial=False):
        """ Predict impedance using an equivalent circuit model

        Parameters
        ----------
        frequencies: ndarray of numeric dtype
        
        max_f: int
            The the maximum frequency of interest for NLEIS

        use_initial: boolean
            If true and the model was previously fit use the initial
            parameters instead

        Returns
        -------
        impedance: ndarray of dtype 'complex128'
            Predicted impedance
        """
        if not isinstance(frequencies, np.ndarray):
            raise TypeError('frequencies is not of type np.ndarray')
        if not (np.issubdtype(frequencies.dtype, np.integer) or
                np.issubdtype(frequencies.dtype, np.floating)):
            raise TypeError('frequencies array should have a numeric ' +
                            f'dtype (currently {frequencies.dtype})')

        f1 =frequencies
        mask = np.array(frequencies)<max_f
        f2 = frequencies[mask]
        
        


        if self._is_fit() and not use_initial:
            x1,x2 = wrappedImpedance(self.circuit_1, self.constants_1,self.circuit_2,self.constants_2,f1,f2,self.parameters_)
            
            return x1,x2
        else:
            warnings.warn("Simulating circuit based on initial parameters")
            x1,x2 = wrappedImpedance(self.circuit_1, self.constants_1,self.circuit_2,self.constants_2,f1,f2,self.initial_guess)
            return(x1,x2)
    def get_param_names(self,circuit,constants):
        """ Converts circuit string to names and units """

        # parse the element names from the circuit string
        names = circuit.replace('t', '').replace('(', '').replace(')', '')#edit

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
        """ Defines the pretty printing of the circuit"""

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

        names1, units1= self.get_param_names(self.circuit_1,self.constants_1)
        names2, units2 = self.get_param_names(self.circuit_2,self.constants_2)

        to_print += '\nEIS Initial guesses:\n'
        p1, p2 = individual_parameters(self.circuit_1,self.initial_guess,self.constants_1,self.constants_2)
        for name, unit, param in zip(names1, units1, p1):
            to_print += '  {:>5} = {:.2e} [{}]\n'.format(name, param, unit)
        to_print += '\nNLEIS Initial guesses:\n'
        for name, unit, param in zip(names2, units2, p2):
            to_print += '  {:>5} = {:.2e} [{}]\n'.format(name, param, unit)
        if self._is_fit():
            params1, confs1 = self.p1, self.conf1
            to_print += '\nEIS Fit parameters:\n'
            for name, unit, param, conf in zip(names1, units1, params1, confs1):
                to_print += '  {:>5} = {:.2e}'.format(name, param)
                to_print += '  (+/- {:.2e}) [{}]\n'.format(conf, unit)
            params2, confs2 = self.p2, self.conf2
            to_print += '\nNLEIS Fit parameters:\n'
            for name, unit, param, conf in zip(names2, units2, params2, confs2):
                to_print += '  {:>5} = {:.2e}'.format(name, param)
                to_print += '  (+/- {:.2e}) [{}]\n'.format(conf, unit)

        return to_print
    def extract(self):
        """ extract the printing of the circuit"""
    
        names1, units1 = self.get_param_names(self.circuit_1,self.constants_1)
        dic1={}
        if self._is_fit():
            params1 = self.p1
            for name, param in zip(names1, params1):
                dic1[name] = param
                
        names2, units2 = self.get_param_names(self.circuit_2,self.constants_2)
        dic2={}
        if self._is_fit():
            params2 = self.p2
            
            for name, param in zip(names2, params2):
                dic2[name] = param
    
        return dic1,dic2
    def plot(self, ax=None, f_data=None, Z1_data =None, Z2_data= None, kind='nyquist', max_f = 10, **kwargs):
        """ visualizes the model and optional data as a nyquist,
            bode, or altair (interactive) plots

        Parameters
        ----------
        ax: matplotlib.axes
            axes to plot on
        f_data: np.array of type float
            Frequencies of input data (for Bode plots)
        Z_data: np.array of type complex
            Impedance data to plot
        kind: {'altair', 'nyquist', 'bode'}
            type of plot to visualize

        Other Parameters
        ----------------
        **kwargs : optional
            If kind is 'nyquist' or 'bode', used to specify additional
             `matplotlib.pyplot.Line2D` properties like linewidth,
             line color, marker color, and labels.
            If kind is 'altair', used to specify nyquist height as `size`

        Returns
        -------
        ax: matplotlib.axes
            axes of the created nyquist plot
        """

        if kind == 'nyquist':
            if ax is None:
                _, ax = plt.subplots(1,2,figsize=(12, 6))

            if Z1_data is not None:
                ax[0] = plot_nyquist(Z1_data, ls='', marker='s', ax=ax[0], **kwargs)
            if Z2_data is not None:
                ax[1] = plot_nyquist(Z2_data, units='Ohms/A', ls='', marker='s', ax=ax[1], **kwargs)
            if self._is_fit():
                if f_data is not None:
                    f_pred = f_data
                else:
                    f_pred = np.logspace(5, -3)
                    
                Z1_fit,Z2_fit = self.predict(f_pred,max_f=max_f)
                ax[0] = plot_nyquist(Z1_fit, ls='-', marker='', ax=ax[0], **kwargs)
                ax[1] = plot_nyquist(Z2_fit,units='Ohms/A', ls='-', marker='', ax=ax[1], **kwargs)
            ax[0].legend(['data','fit'])
            ax[1].legend(['data','fit'])
            return ax
        elif kind == 'bode':
            if ax is None:
                _, ax = plt.subplots(2,2, figsize=(8, 8))

            if f_data is not None:
                f_pred = f_data
            else:
                f_pred = np.logspace(5, -3)

            if Z1_data is not None:
                if f_data is None:
                    raise ValueError('f_data must be specified if' +
                                     ' Z_data for a Bode plot')
                ax[:,0] = plot_bode(f_data, Z1_data, ls='', marker='s',
                               axes=ax[:,0], **kwargs)
                # ax[:,0].set_xlabel('')
            if Z2_data is not None:
                if f_data is None:
                    raise ValueError('f_data must be specified if' +
                                     ' Z_data for a Bode plot')
                f2 = f_data[np.array(f_data)<max_f]
                ax[:,1] = plot_bode(f2, Z2_data,units='Ohms/A', ls='', marker='s',
                               axes=ax[:,1], **kwargs)

            if self._is_fit():
                Z1_fit,Z2_fit = self.predict(f_pred,max_f=max_f)
                f1 = f_data
                f2 = f_data[np.array(f_data)<max_f]

                ax[:,0] = plot_bode(f1, Z1_fit, ls='-', marker='o',
                               axes=ax[:,0], **kwargs)
                ax[:,1] = plot_bode(f2, Z2_fit,units='Ohms/A', ls='-', marker='o',
                               axes=ax[:,1], **kwargs)
            ax[0,0].set_ylabel(r'$|Z_{1}(\omega)|$ ' +
                      '$[{}]$'.format('Ohms'), fontsize=20)
            ax[1,0].set_ylabel(r'$-\phi_{Z_{1}}(\omega)$ ' + r'$[^o]$', fontsize=20)
            ax[0,1].set_ylabel(r'$|Z_{2}(\omega)|$ ' +
                      '$[{}]$'.format('Ohms/A'), fontsize=20)
            ax[1,1].set_ylabel(r'$-\phi_{Z_{2}}(\omega)$ ' + r'$[^o]$', fontsize=20)
            ax[0,0].legend(['Data','Fit'],fontsize=20)
            ax[0,1].legend(['Data','Fit'],fontsize=20)
            ax[1,0].legend(['Data','Fit'],fontsize=20)
            ax[1,1].legend(['Data','Fit'],fontsize=20)
            return ax
        elif kind == 'altair':
            plot_dict_1 = {}
            plot_dict_2 = {}


            if Z1_data is not None and Z2_data is not None and f_data is not None:
                plot_dict_1['data'] = {'f': f_data, 'Z': Z1_data}
                plot_dict_2['data'] = {'f': f_data[np.array(f_data)<max_f], 'Z': Z2_data}

            if self._is_fit():
                if f_data is not None:
                    
                    f_pred = f_data
                    
                else:
                   
                    f_pred = np.logspace(5, -3)

                if self.name is not None:
                    name = self.name
                else:
                    name = 'fit'
                Z1_fit,Z2_fit = self.predict(f_pred,max_f=max_f)

                plot_dict_1[name] = {'f': f_pred, 'Z': Z1_fit, 'fmt': '-'}
                plot_dict_2[name] = {'f': f_pred[np.array(f_pred)<max_f], 'Z': Z2_fit, 'fmt': '-'}
            

            chart1 = plot_altair(plot_dict_1,units = 'Ω', **kwargs)
            chart2 = plot_altair(plot_dict_2,units = 'Ω/A', **kwargs)

            return chart1,chart2
        else:
            raise ValueError("Kind must be one of 'altair'," +
                             f"'nyquist', or 'bode' (received {kind})")
    def save(self, filepath):
        """ Exports a model to JSON

        Parameters
        ----------
        filepath: str
            Destination for exporting model object
        """

        model_string_1 = self.circuit_1
        model_string_2 = self.circuit_2

        
        model_name = self.name

        initial_guess = self.initial_guess

        if self._is_fit():
            parameters_ = list(self.parameters_)
            model_conf_ = list(self.conf_)

            data_dict = {"Name": model_name,
                         "Circuit String 1": model_string_1,
                         "Circuit String 2": model_string_2,
                         "Initial Guess": initial_guess,
                         "Constants": self.constants,
                         "Fit": True,
                         "Parameters": parameters_,
                         "Confidence": model_conf_,
                         }
        else:
            data_dict = {"Name": model_name,
                         "Circuit String 1": model_string_1,
                         "Circuit String 2": model_string_2,
                         "Initial Guess": initial_guess,
                         "Constants": self.constants,
                         "Fit": False}

        with open(filepath, 'w') as f:
            json.dump(data_dict, f)


    def load(self, filepath, fitted_as_initial=False):
        """ Imports a model from JSON

        Parameters
        ----------
        filepath: str
            filepath to JSON file to load model from

        fitted_as_initial: bool
            If true, loads the model's fitted parameters
            as initial guesses

            Otherwise, loads the model's initial and
            fitted parameters as a completed model
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

        print(self.circuit_1)
        print(self.circuit_2)

        self.constants = model_constants
        self.name = model_name

        if json_data["Fit"]:
            if fitted_as_initial:
                self.initial_guess = np.array(json_data['Parameters'])
            else:
                self.parameters_ = np.array(json_data["Parameters"])
                self.conf_ = np.array(json_data["Confidence"])
    

class NLEISCustomCircuit(BaseCircuit):
    def __init__(self, circuit='', **kwargs):
        """ Constructor for a customizable equivalent circuit model

        Parameters
        ----------
        initial_guess: numpy array
            Initial guess of the circuit values

        circuit: string
            A string that should be interpreted as an equivalent circuit

        Notes
        -----
        A custom circuit is defined as a string comprised of elements in series
        (separated by a `-`) and elements in parallel (grouped as (x,y)).
        Each element can be appended with an integer (e.g. R0) or an underscore
        and an integer (e.g. CPE_1) to make keeping track of multiple elements
        of the same type easier.

        Example:
            Randles circuit is given by 'R0-p(R1-Wo1,C1)'

        """

        super().__init__(**kwargs)
        self.circuit = circuit.replace(" ", "")

        circuit_len = calculateCircuitLength(self.circuit)

        if len(self.initial_guess) + len(self.constants) != circuit_len:
            raise ValueError('The number of initial guesses ' +
                             f'({len(self.initial_guess)}) + ' +
                             'the number of constants ' +
                             f'({len(self.constants)})' +
                             ' must be equal to ' +
                             f'the circuit length ({circuit_len})')
    def fit(self, frequencies, impedance, bounds=None,
            weight_by_modulus=False, **kwargs):
        """ Fit the circuit model

        Parameters
        ----------
        frequencies: numpy array
            Frequencies

        impedance: numpy array of dtype 'complex128'
            Impedance values to fit

        bounds: 2-tuple of array_like, optional
            Lower and upper bounds on parameters. Defaults to bounds on all
            parameters of 0 and np.inf, except the CPE alpha
            which has an upper bound of 1

        weight_by_modulus : bool, optional
            Uses the modulus of each data (|Z|) as the weighting factor.
            Standard weighting scheme when experimental variances are
            unavailable. Only applicable when global_opt = False

        kwargs :
            Keyword arguments passed to
            impedance.models.circuits.fitting.circuit_fit,
            and subsequently to scipy.optimize.curve_fit
            or scipy.optimize.basinhopping

        Returns
        -------
        self: returns an instance of self

        """

            
        frequencies = np.array(frequencies, dtype=float)
        impedance = np.array(impedance, dtype=complex)

        if len(frequencies) != len(impedance):
            raise TypeError('length of frequencies and impedance do not match')

        if self.initial_guess != []:
            parameters, conf = circuit_fit(frequencies, impedance,
                                           self.circuit, self.initial_guess,
                                           constants=self.constants,
                                           bounds=bounds,
                                           weight_by_modulus=weight_by_modulus,
                                           **kwargs)
            self.parameters_ = parameters
            if conf is not None:
                self.conf_ = conf
        else:
            raise ValueError('No initial guess supplied')

        return self
    def predict(self, frequencies, use_initial=False):
        """ Predict impedance using an equivalent circuit model

        Parameters
        ----------
        frequencies: array-like of numeric type
        use_initial: boolean
            If true and the model was previously fit use the initial
            parameters instead

        Returns
        -------
        impedance: ndarray of dtype 'complex128'
            Predicted impedance at each frequency
        """
        frequencies = np.array(frequencies, dtype=float)

        if self._is_fit() and not use_initial:
            return eval(buildCircuit(self.circuit, frequencies,
                                     *self.parameters_,
                                     constants=self.constants, eval_string='',
                                     index=0)[0],
                        circuit_elements)
        else:
            warnings.warn("Simulating circuit based on initial parameters")
            return eval(buildCircuit(self.circuit, frequencies,
                                     *self.initial_guess,
                                     constants=self.constants, eval_string='',
                                     index=0)[0],
                        circuit_elements)
    def get_param_names(self):
        """ Converts circuit string to names and units """

        # parse the element names from the circuit string
        names = self.circuit.replace('t', '').replace('(', '').replace(')', '')#edit

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
        """ extract the printing of the circuit"""

        names, units = self.get_param_names()
        dic={}
        if self._is_fit():
            params, confs = self.parameters_, self.conf_
            
            for name, unit, param, conf in zip(names, units, params, confs):
                dic[name] = param

        return dic

    
