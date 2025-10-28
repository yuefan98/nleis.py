import warnings
import json  # noqa: F401
import os

import numpy as np
import matplotlib.pyplot as plt  # noqa: F401
import pytest

from nleis.nleis import EISandNLEIS, NLEISCustomCircuit  # noqa: F401

from nleis.data_processing import data_truncation

test_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(test_dir, '../data')

# supress warnings for cleaner test output
warnings.filterwarnings("ignore")

# # get example data
# # The example is shown in "Getting Started" page

frequencies = np.loadtxt(os.path.join(data_dir, 'freq_30a.txt'))
Z1 = np.loadtxt(os.path.join(data_dir, 'Z1s_30a.txt')).view(complex)[1]
Z2 = np.loadtxt(os.path.join(data_dir, 'Z2s_30a.txt')).view(complex)[1]

f, Z1, Z2, f2_trunc, Z2_trunc = data_truncation(frequencies, Z1, Z2)


def test_parsing():
    circ_str_1 = 'L0-R0-TDS0-TDS1'
    circ_str_2 = 'd(TDSn0,TDSn1)'
    initial_guess = [1e-7, 1e-3,  # L0,RO
                     5e-3, 1e-3, 1e-2, 100, 10, 0.1,
                     # TDS0 + additioal nonlinear parameters
                     1e-3, 1e-3, 1e-3, 1e-2, 1000, 0,
                     # TDS1 + additioal nonlinear parameters
                     ]

    NLEIS_circuit = EISandNLEIS(
        circ_str_1, circ_str_2, initial_guess=initial_guess,
        constants={'TDSn0_2': 10, 'TDSn1_6': 0})

    assert NLEIS_circuit.edited_circuit == 'L0-R0-TDSn0-TDSn1'
    assert NLEIS_circuit.constants_1 == {'TDS0_2': 10}
    assert NLEIS_circuit.constants_2 == {'TDSn0_2': 10, 'TDSn1_6': 0}


def test_EISandNLEIS():
    circ_str_1 = 'L0-R0-TDS0-TDS1'
    circ_str_2 = 'd(TDSn0,TDSn1)'
    initial_guess = [1e-7, 1e-3,  # L0,RO
                     5e-3, 1e-3, 10, 1e-2, 100, 10, 0.1,
                     # TDS0 + additioal nonlinear parameters
                     1e-3, 1e-3, 1e-3, 1e-2, 1000, 0, 0,
                     # TDS1 + additioal nonlinear parameters
                     ]

    NLEIS_circuit = EISandNLEIS(
        circ_str_1, circ_str_2, initial_guess=initial_guess)

    # check get_param_names()
    full_names_EIS, all_units_EIS = NLEIS_circuit.get_param_names(
        circ_str_1, {})

    assert full_names_EIS == ['L0', 'R0', 'TDS0_0', 'TDS0_1', 'TDS0_2',
                              'TDS0_3', 'TDS0_4',
                              'TDS1_0', 'TDS1_1', 'TDS1_2', 'TDS1_3', 'TDS1_4']
    assert all_units_EIS == ['H', 'Ohm', 'Ohms', 'Ohms',
                             'F', 'Ohms', 's', 'Ohms', 'Ohms',
                             'F', 'Ohms', 's']
    full_names_NLEIS, all_units_NLEIS = NLEIS_circuit.get_param_names(
        circ_str_2, {})
    assert full_names_NLEIS == ['TDSn0_0', 'TDSn0_1', 'TDSn0_2', 'TDSn0_3',
                                'TDSn0_4', 'TDSn0_5',
                                'TDSn0_6', 'TDSn1_0', 'TDSn1_1', 'TDSn1_2',
                                'TDSn1_3', 'TDSn1_4', 'TDSn1_5', 'TDSn1_6']
    assert all_units_NLEIS == ['Ohms', 'Ohms', 'F', 'Ohms', 's',
                               '1/V', '-', 'Ohms', 'Ohms', 'F', 'Ohms', 's',
                               '1/V', '-']

    # check _is_fit()
    assert not NLEIS_circuit._is_fit()

    # check complex frequencies raise TypeError
    with pytest.raises(TypeError):
        NLEIS_circuit.predict([0.42, 42 + 42j])

    initial_guess = [1e-3,  # RO
                     5e-3, 1e-3, 10, 1e-2, 100, 10, 0.1,
                     # TDS0 + additioal nonlinear parameters
                     1e-3, 1e-3, 1e-3, 1e-2, 1000, 0,
                     # TDS1 + additioal nonlinear parameters
                     ]
    # check correct assignment of constants
    # when EIS element is supplied
    NLEIS_circuit = EISandNLEIS(
        circ_str_1, circ_str_2, initial_guess=initial_guess,
        constants={'L0': 1e-7, 'TDS1_3': 1})

    assert {'L0': 1e-7, 'TDS1_3': 1} == NLEIS_circuit.constants_1
    assert {'TDSn1_3': 1} == NLEIS_circuit.constants_2

    # check correct assignment of constants
    # when 2nd-NLEIS element is supplied
    NLEIS_circuit = EISandNLEIS(
        circ_str_1, circ_str_2, initial_guess=initial_guess,
        constants={'L0': 1e-7, 'TDSn1_6': 0})
    assert {'L0': 1e-7} == NLEIS_circuit.constants_1
    assert {'TDSn1_6': 0} == NLEIS_circuit.constants_2

    # raise ValueError if the length of the frequency
    # does not matches with the length of the impedance for EIS
    with pytest.raises(ValueError):
        NLEIS_circuit.fit(f[f < 1], Z1, Z2)

    # raise ValueError if no initial_guess is supplied
    with pytest.raises(ValueError):
        NLEIS_circuit = EISandNLEIS(
            circ_str_1, circ_str_2, initial_guess=initial_guess,
            constants={'L0': 1e-7, 'TDSn1_6': 0})
        NLEIS_circuit.fit(f[f < 1], Z1, Z2)

    # check non-number initial guess raise TypeError
    with pytest.raises(TypeError):
        NLEIS_circuit = EISandNLEIS(
            circ_str_1, circ_str_2, initial_guess=['a'],
            constants={'L0': 1e-7, 'TDSn1_6': 0})

    # check mismatched EIS and 2nd-NLEIS circuit
    # raise TypeError
    with pytest.raises(TypeError):
        NLEIS_circuit = EISandNLEIS(
            circ_str_1, 'd(TDS0,TDSn2)', initial_guess=initial_guess,
            constants={'L0': 1e-7, 'TDSn1_6': 0})

    # check wrong constants raise
    # raise ValueError
    with pytest.raises(ValueError):
        NLEIS_circuit = EISandNLEIS(
            circ_str_1, circ_str_2, initial_guess=initial_guess,
            constants={'L0': 1e-7, 'TDSn2_6': 0})

    # check that constants number beyond the range
    # of allowed number of parameters
    # raise ValueError
    with pytest.raises(ValueError):
        NLEIS_circuit = EISandNLEIS(
            circ_str_1, circ_str_2, initial_guess=initial_guess,
            constants={'L0': 1e-7, 'TDSn1_7': 0})
    # check that constants number beyond the range
    # of allowed number of parameters
    # raise ValueError
    with pytest.raises(ValueError):
        NLEIS_circuit = EISandNLEIS(
            circ_str_1, circ_str_2, initial_guess=initial_guess,
            constants={'L2': 1e-7, 'TDSn1_6': 0})

    # check incorrect element pair raise TypeError
    with pytest.raises(TypeError):
        NLEIS_circuit = EISandNLEIS(
            circ_str_1, circ_str_1, initial_guess=initial_guess,
            constants={'L0': 1e-7, 'TDSn1_6': 0})

    # check that constants either circ_str_1 or circ_str_2
    # cannot be empty/ raise ValueError if empty
    # check missing circuit_2 raise ValueError
    with pytest.raises(ValueError):
        NLEIS_circuit = EISandNLEIS(
            circ_str_1, circ_str_2='', initial_guess=initial_guess,
            constants={'L0': 1e-7, 'TDSn1_6': 0})

    # check that wrong length of initial_guess
    # with respect to the length of the circuit raise ValueError

    with pytest.raises(ValueError):
        NLEIS_circuit = EISandNLEIS(
            circ_str_1, circ_str_2, initial_guess=[1],
            constants={'L0': 1e-7, 'TDSn1_6': 0})

    # check that wrong constants input raise ValueError
    # when the wrong constants is applied for EIS_circuit

    with pytest.raises(ValueError):
        NLEIS_circuit = EISandNLEIS(
            circ_str_1, circ_str_2, initial_guess=initial_guess,
            constants={'L0': 1e-7, 'TDS1_10': 0})


def test_eq():
    NLEIS_circuit = NLEISCustomCircuit()
    simul_circuit = EISandNLEIS()

    with pytest.raises(TypeError):
        simul_circuit == NLEIS_circuit


def test_EISandNLEIS_fitting():
    circ_str_1 = 'L0-R0-TDS0-TDS1'
    circ_str_2 = 'd(TDSn0,TDSn1)'

    # Test example shown "Getting Started" page
    # Note: solution to the solid state diffuion might
    # converge to different value given the exact
    # envrionment used (i.e. large confidence interval)

    # The test framework here can be improved in the future
    # with better and stable initial guess
    results = [2.76862889e-07, 9.27338912e-03, 2.57626033e-02, 6.09369447e-03,
               6.91681283e+00, 1.23574283e-04, 5.13935770e+00, 2.54622037e+01,
               1.37710342e-01, 2.45544428e-02, 2.65941949e-03, 7.05861025e-02,
               1.85837897e+02, 3.20561842e+09, 1.87142864e+00, 3.66300641e-03]

    initial_guess = results

    # initial_guess
    # L0,RO
    # TDS0 + additioal nonlinear parameters
    # TDS1 + additioal nonlinear parameters

    NLEIS_circuit = EISandNLEIS(
        circ_str_1, circ_str_2, initial_guess=initial_guess)

    # test predict with initial_guess
    Z1_fit, Z2_fit = NLEIS_circuit.predict(f, max_f=10)
    assert np.allclose(Z1, Z1_fit, rtol=1e-2, atol=1e-2)
    assert np.allclose(Z2_trunc, Z2_fit, rtol=1e-2, atol=1e-2)

    # test fitting
    NLEIS_circuit.fit(f, Z1, Z2, max_f=10)
    p = NLEIS_circuit.parameters_
    print(p)

    assert np.allclose(p, results)

    # test the extract method
    dict1, dict2 = NLEIS_circuit.extract()
    assert list(dict1.keys()) == ['L0', 'R0',
                                  'TDS0_0', 'TDS0_1', 'TDS0_2', 'TDS0_3',
                                  'TDS0_4',
                                  'TDS1_0', 'TDS1_1', 'TDS1_2',
                                  'TDS1_3', 'TDS1_4']
    assert list(dict2.keys()) == ['TDSn0_0', 'TDSn0_1', 'TDSn0_2', 'TDSn0_3',
                                  'TDSn0_4', 'TDSn0_5',
                                  'TDSn0_6', 'TDSn1_0', 'TDSn1_1', 'TDSn1_2',
                                  'TDSn1_3', 'TDSn1_4', 'TDSn1_5', 'TDSn1_6']

    # test plotting
    # kind = {'nyquist', 'bode'} should return a plt.Axes() object
    _, ax = plt.subplots(1, 2)
    assert isinstance(NLEIS_circuit.plot(
        ax, None, Z1, Z2, kind='nyquist'), type(ax))
    assert isinstance(NLEIS_circuit.plot(
        None, f, Z1, Z2, kind='nyquist'), type(ax))
    _, axes = plt.subplots(2, 2)
    assert isinstance(NLEIS_circuit.plot(
        axes, f, Z1, Z2, kind='bode')[0], type(ax))
    assert isinstance(NLEIS_circuit.plot(
        None, f, Z1, Z2, kind='bode')[0], type(ax))

    # check altair plotting with a fit circuit
    chart1, chart2 = NLEIS_circuit.plot(f_data=f, Z1_data=Z1, Z2_data=Z2,
                                        max_f=10, kind='altair')

    datasets = json.loads(chart1.to_json())['datasets']
    for dataset in datasets.keys():
        assert len(datasets[dataset]) == len(Z1)

    datasets = json.loads(chart2.to_json())['datasets']
    for dataset in datasets.keys():
        assert len(datasets[dataset]) == len(Z2_trunc)

    # incorrect kind raises a ValueError
    with pytest.raises(ValueError):
        NLEIS_circuit.plot(None, f, Z1, Z2, kind='SomethingElse')

    # test optimization with negative log-likelihood
    initial_guess = [9.81387341e-08, 1.34551661e-02, 2.52404567e-02,
                     5.06142381e-03,
                     8.82333612e+00, 8.80981845e-05, 3.59757749e+00,
                     1.22590097e+01,
                     8.74790184e-02, 2.09035936e-02, 1.13816141e-03,
                     8.13599509e-01,
                     1.71140213e+02, 2.77968560e+09, 1.02284630e+00,
                     6.38829808e-03]
    NLEIS_circuit = EISandNLEIS(
        circ_str_1, circ_str_2, initial_guess=initial_guess)

    NLEIS_circuit.fit(f, Z1, Z2, opt='neg')
    p2 = NLEIS_circuit.parameters_
    results2 = [9.81387329e-08, 1.34551659e-02, 2.52509590e-02, 5.14440732e-03,
                8.82333612e+00, 8.80981835e-05, 3.59757749e+00, 1.22590097e+01,
                8.74790233e-02, 2.09083711e-02, 1.13829972e-03, 8.13599522e-01,
                1.71140213e+02, 2.77968560e+09, 1.02284629e+00, 6.38830393e-03]

    assert np.allclose(p2, results2, rtol=1e-3, atol=1e-3)


def test_NLEISCustomCircuit():
    circ_str = 'd(TDSn0,TDSn1)'
    initial_guess = [
        5e-3, 1e-3, 10, 1e-2, 100, 10, 0.1,
        # TDS0 + additioal nonlinear parameters
        1e-3, 1e-3, 1e-3, 1e-2, 1000, 0, 0,
        # TDS1 + additioal nonlinear parameters
    ]

    NLEIS_circuit = NLEISCustomCircuit(
        circ_str, initial_guess=initial_guess)

    assert not NLEIS_circuit._is_fit()

    # check get_param_names()
    full_names_NLEIS, all_units_NLEIS = NLEIS_circuit.get_param_names()
    assert full_names_NLEIS == ['TDSn0_0', 'TDSn0_1', 'TDSn0_2', 'TDSn0_3',
                                'TDSn0_4', 'TDSn0_5',
                                'TDSn0_6', 'TDSn1_0', 'TDSn1_1', 'TDSn1_2',
                                'TDSn1_3', 'TDSn1_4', 'TDSn1_5', 'TDSn1_6']
    assert all_units_NLEIS == ['Ohms', 'Ohms', 'F', 'Ohms', 's',
                               '1/V', '-', 'Ohms', 'Ohms', 'F', 'Ohms', 's',
                               '1/V', '-']

    # check _is_fit()
    assert not NLEIS_circuit._is_fit()

    # check complex frequencies raise TypeError
    with pytest.raises(TypeError):
        NLEIS_circuit.predict([0.42, 42 + 42j])

    # test is_fit method
    NLEIS_circuit.fit(f, Z2)
    assert NLEIS_circuit._is_fit()

    # test extract method
    dict = NLEIS_circuit.extract()
    assert list(dict.keys()) == ['TDSn0_0', 'TDSn0_1', 'TDSn0_2', 'TDSn0_3',
                                 'TDSn0_4', 'TDSn0_5',
                                 'TDSn0_6', 'TDSn1_0', 'TDSn1_1', 'TDSn1_2',
                                 'TDSn1_3', 'TDSn1_4', 'TDSn1_5', 'TDSn1_6']

    # test plotting
    # kind = {'nyquist', 'bode'} should return a plt.Axes() object
    _, ax = plt.subplots()
    assert isinstance(NLEIS_circuit.plot(
        ax, None, Z2, kind='nyquist'), type(ax))
    assert isinstance(NLEIS_circuit.plot(
        None, f, Z2, kind='nyquist'), type(ax))
    _, axes = plt.subplots(1, 2)
    assert isinstance(NLEIS_circuit.plot(
        axes, f, Z2, kind='bode')[0], type(ax))
    assert isinstance(NLEIS_circuit.plot(
        None, f, Z2, kind='bode')[0], type(ax))

    # check altair plotting with a fit circuit
    chart = NLEIS_circuit.plot(f_data=f, Z2_data=Z2, max_f=10,
                               kind='altair')

    datasets = json.loads(chart.to_json())['datasets']
    for dataset in datasets.keys():
        assert len(datasets[dataset]) == len(Z2_trunc)

    # incorrect kind raises a ValueError
    with pytest.raises(ValueError):
        NLEIS_circuit.plot(None, f, Z2, kind='SomethingElse')
