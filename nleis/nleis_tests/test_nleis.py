import json  # noqa: F401
import os

import numpy as np
import matplotlib.pyplot as plt  # noqa: F401
import pytest

from nleis.nleis import EISandNLEIS, NLEISCustomCircuit  # noqa: F401

from nleis.nleis_fitting import data_processing

test_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(test_dir, '../data')

# # get example data
# # The example is shown in "Getting Started" page

frequencies = np.loadtxt(os.path.join(data_dir, 'freq_30a.txt'))
Z1 = np.loadtxt(os.path.join(data_dir, 'Z1s_30a.txt')).view(complex)[1]
Z2 = np.loadtxt(os.path.join(data_dir, 'Z2s_30a.txt')).view(complex)[1]

f, Z1, Z2, f2_trunc, Z2_trunc = data_processing(frequencies, Z1, Z2)


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
                               '1/V', '', 'Ohms', 'Ohms', 'F', 'Ohms', 's',
                               '1/V', '']

    # check _is_fit()
    assert not NLEIS_circuit._is_fit()

    # check complex frequencies raise TypeError
    with pytest.raises(TypeError):
        NLEIS_circuit.predict([0.42, 42 + 42j])


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

    # check get_param_names()
    full_names_NLEIS, all_units_NLEIS = NLEIS_circuit.get_param_names()
    assert full_names_NLEIS == ['TDSn0_0', 'TDSn0_1', 'TDSn0_2', 'TDSn0_3',
                                'TDSn0_4', 'TDSn0_5',
                                'TDSn0_6', 'TDSn1_0', 'TDSn1_1', 'TDSn1_2',
                                'TDSn1_3', 'TDSn1_4', 'TDSn1_5', 'TDSn1_6']
    assert all_units_NLEIS == ['Ohms', 'Ohms', 'F', 'Ohms', 's',
                               '1/V', '', 'Ohms', 'Ohms', 'F', 'Ohms', 's',
                               '1/V', '']

    # check _is_fit()
    assert not NLEIS_circuit._is_fit()

    # check complex frequencies raise TypeError
    with pytest.raises(TypeError):
        NLEIS_circuit.predict([0.42, 42 + 42j])


def test_fitting():
    circ_str_1 = 'L0-R0-TDS0-TDS1'
    circ_str_2 = 'd(TDSn0,TDSn1)'

    # Test example shown "Getting Started" page
    # Note: solution to the solid state diffuion might
    # converge to different value given the exact
    # envrionment used (i.e. large confidence interval)

    # The test framework here can be improved in the future
    # with better and stable initial guess
    results = [9.81368514e-08, 1.34551972e-02, 2.52387276e-02, 5.06176242e-03,
               8.82244297e+00, 8.70692162e-05, 3.55536976e+00, 1.22576118e+01,
               8.75169434e-02, 2.09045802e-02, 1.13804384e-03, 8.13658287e-01,
               1.83783329e+02, 3.20554700e+09, 1.02277512e+00, 6.39228801e-03]
    initial_guess = results

    # initial_guess
    # L0,RO
    # TDS0 + additioal nonlinear parameters
    # TDS1 + additioal nonlinear parameters

    NLEIS_circuit = EISandNLEIS(
        circ_str_1, circ_str_2, initial_guess=initial_guess)

    NLEIS_circuit.fit(f, Z1, Z2)
    p = NLEIS_circuit.parameters_

    assert np.allclose(p, results)
