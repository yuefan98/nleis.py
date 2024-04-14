import json
import os

import numpy as np
import matplotlib.pyplot as plt
import pytest

from impedance.models.circuits import BaseCircuit, CustomCircuit, Randles
from impedance.models.nleis import EISandNLEIS, NLEISCustomCircuit

# get example data
frequencies = np.loadtxt('data/freq_30a.txt')
Z1 = np.loadtxt('data/Z1s_30a.txt').view(complex)[1]
Z2 = np.loadtxt('data/Z2s_30a.txt').view(complex)[1]



# def test_EISandNLEIS():
#     initial_guess = [0.01, 0.02, 50]

#     # __init__()
#     # check initial_guess is loaded in correctly
#     base_circuit = EISandNLEIS(initial_guess=initial_guess)
#     assert base_circuit.initial_guess == initial_guess

#     # improper input_guess types raise an TypeError
#     with pytest.raises(TypeError):
#         r = BaseCircuit(initial_guess=['hi', 0.1])

#     # __eq__()
#     # incorrect comparisons raise a TypeError
#     with pytest.raises(TypeError):
#         r = BaseCircuit(initial_guess=[.01, .005, .1, .0001, 200])
#         r == 8

#     # fit()
#     with pytest.raises(TypeError):
#         r = BaseCircuit(initial_guess=[.01, .005, .1, .0001, 200])
#         r.fit(np.array([42 + 42j]), [])  # frequencies are complex

#     with pytest.raises(TypeError):
#         r = BaseCircuit(initial_guess=[.01, .005, .1, .0001, 200])
#         r.fit(np.array([42, 4.2]), np.array([42 + 42j]))  # mismatched lengths

#     # plot()
#     # kind = {'nyquist', 'bode'} should return a plt.Axes() object
#     _, ax = plt.subplots()
#     assert isinstance(base_circuit.plot(ax, None, Z, kind='nyquist'), type(ax))
#     assert isinstance(base_circuit.plot(None, f, Z, kind='nyquist'), type(ax))
#     _, axes = plt.subplots(nrows=2)
#     assert isinstance(base_circuit.plot(axes, f, Z, kind='bode')[0], type(ax))
#     assert isinstance(base_circuit.plot(None, f, Z, kind='bode')[0], type(ax))

#     # incorrect kind raises a ValueError
#     with pytest.raises(ValueError):
#         base_circuit.plot(None, f, Z, kind='SomethingElse')


# def test_Randles():
#     randles = Randles(initial_guess=[.01, .005, .01, 200, .1])
#     randlesCPE = Randles(initial_guess=[.01, .05, .01, 200, .1, 0.9], CPE=True)
#     with pytest.raises(ValueError):
#         randlesCPE = Randles([.01, 200])  # incorrect initial guess length
#     randles.fit(f[np.imag(Z) < 0], Z[np.imag(Z) < 0])
#     randlesCPE.fit(f[np.imag(Z) < 0], Z[np.imag(Z) < 0])

#     # compare with known fit parameters
#     np.testing.assert_almost_equal(randles.parameters_,
#                                    np.array([1.86235717e-02, 1.16804085e-02,
#                                              6.27121224e-02, 2.21232935e+02,
#                                              1.17171440e+00]), decimal=2)

#     # compare with known impedance predictions
#     assert np.isclose(randles.predict(np.array([10.0])),
#                       complex(0.0251618, -0.00601304))

#     # check altair plotting with a fit circuit
#     chart = randles.plot(f_data=f, Z_data=Z)
#     datasets = json.loads(chart.to_json())['datasets']
#     for dataset in datasets.keys():
#         assert len(datasets[dataset]) == len(Z)

#     # plot() with fitted model
#     # check defaults work if no frequency data is passed
#     chart = randles.plot(Z_data=Z)

#     # bode plots
#     randles.plot(f_data=f, Z_data=Z, kind='bode')
#     randles.plot(kind='bode')
#     with pytest.raises(ValueError):
#         randles.plot(Z_data=Z, kind='bode')  # missing f_data

#     # nyquist plots
#     randles.plot(f_data=f, Z_data=Z, kind='nyquist')
#     randles.plot(Z_data=Z, kind='nyquist')

#     # check equality comparisons work
#     randles1 = Randles(initial_guess=[.01, .005, .0001, 200, .1])
#     randles2 = Randles(initial_guess=[.01, .005, .0001, 200, .1])
#     assert randles1 == randles2

#     randles1.fit(f[np.imag(Z) < 0], Z[np.imag(Z) < 0])
#     assert randles1 != randles2

#     randles2.fit(f[np.imag(Z) < 0], Z[np.imag(Z) < 0])
#     assert randles1 == randles2

#     randles2.fit(f, Z)
#     assert randles1 != randles2


def test_EISandNLEIS():
    circ_str_1 = 'L0-R0-TDS0-TDS1'
    circ_str_2 = 'd(TDSn0,TDSn1)'
    initial_guess = [1e-7,1e-3 # L0,RO
                    ,5e-3,1e-3,10,1e-2,100,10,0.1 ## TDS0 + additioal nonlinear parameters
                    ,1e-3,1e-3,1e-3,1e-2,1000,0,0 ## TDS1 + additioal nonlinear parameters
                    ]

    NLEIS_circuit = EISandNLEIS(circ_str_1,circ_str_2,initial_guess=initial_guess)

    # check get_param_names()
    full_names_EIS, all_units_EIS = NLEIS_circuit.get_param_names(circ_str_1,{})
    assert full_names_EIS == ['L0', 'R0', 'TDS0_0', 'TDS0_1','TDS0_2','TDS0_3','TDS0_4',
                              'TDS1_0','TDS1_1','TDS1_2','TDS1_3','TDS1_4']
    assert all_units_EIS == ['H','Ohm','Ohms', 'Ohms', 'F','Ohms','s','Ohms', 'Ohms', 'F','Ohms','s']
    full_names_NLEIS, all_units_NLEIS = NLEIS_circuit.get_param_names(circ_str_2,{})
    assert full_names_NLEIS == ['TDSn0_0', 'TDSn0_1','TDSn0_2','TDSn0_3','TDSn0_4','TDSn0_5','TDSn0_6'
                                ,'TDSn1_0','TDSn1_1','TDSn1_2','TDSn1_3','TDSn1_4','TDSn1_5','TDSn1_6']
    assert all_units_NLEIS == ['Ohms', 'Ohms', 'F','Ohms','s','1/V','','Ohms', 'Ohms', 'F','Ohms','s','1/V','']

#     # check _is_fit()
#     assert not custom_circuit._is_fit()

#     # check predictions from initial_guesses
#     high_f = np.array([1e9])
#     assert np.isclose(np.real(custom_circuit.predict(high_f)[0]),
#                       initial_guess[0])

#     # check complex frequencies raise TypeError
#     with pytest.raises(TypeError):
#         custom_circuit.predict([0.42, 42 + 42j])

#     # __str()__
#     initial_guess = [.01, .005, .1]
#     custom_string = 'R0-p(R1,C1)'
#     custom_circuit = CustomCircuit(initial_guess=initial_guess,
#                                    circuit=custom_string)

#     assert str(custom_circuit) == \
#         '\nCircuit string: R0-p(R1,C1)\n' + \
#         'Fit: False\n' + \
#         '\nInitial guesses:\n' + \
#         '     R0 = 1.00e-02 [Ohm]\n' + \
#         '     R1 = 5.00e-03 [Ohm]\n' + \
#         '     C1 = 1.00e-01 [F]\n'

#     custom_circuit.fit(f, Z)
#     assert custom_circuit._is_fit()
#     custom_circuit.plot(f_data=f, Z_data=Z)

#     # constants and _ in circuit and no name
#     circuit = 'R_0-p(R_1,C_1)-Wo_1'
#     constants = {'R_0': 0.02, 'Wo_1_1': 200}
#     initial_guess = [.005, .1, .001]
#     custom_circuit = CustomCircuit(initial_guess=initial_guess,
#                                    constants=constants, circuit=circuit,
#                                    name='Test')

#     assert str(custom_circuit) == \
#         '\nName: Test\n' + \
#         'Circuit string: R_0-p(R_1,C_1)-Wo_1\n' + \
#         'Fit: False\n' + \
#         '\nConstants:\n' + \
#         '    R_0 = 2.00e-02 [Ohm]\n' + \
#         '  Wo_1_1 = 2.00e+02 [sec]\n' + \
#         '\nInitial guesses:\n' + \
#         '    R_1 = 5.00e-03 [Ohm]\n' + \
#         '    C_1 = 1.00e-01 [F]\n' + \
#         '  Wo_1_0 = 1.00e-03 [Ohm]\n'

#     # incorrect number of initial guesses
#     with pytest.raises(ValueError):
#         initial_guess = [.01, .005, .1, .005, .1, .001, 200]
#         custom_string = 'R0-p(R1,CPE1)-p(R1,C1)-Wo1'
#         custom_circuit = CustomCircuit(initial_guess=initial_guess,
#                                        circuit=custom_string)

#     # no initial guesses supplied before fitting
#     with pytest.raises(ValueError):
#         custom_circuit = CustomCircuit()
#         custom_circuit.fit(f, Z)

#     # incorrect circuit element in circuit
#     with pytest.raises(ValueError):
#         custom_circuit = CustomCircuit('R0-NotAnElement', initial_guess=[1, 2])

#     # test single element circuit
#     initial_guess = [1]
#     custom_string = 'R0'
#     custom_circuit = CustomCircuit(initial_guess=initial_guess,
#                                    circuit=custom_string)
#     custom_circuit.fit([1, 2, 3], [4, 4, 4])
#     assert custom_circuit.parameters_[0] == 4

#     # space in circuit string
#     circuit = circuit = 'R0-p(R1, C1)'
#     initial_guess = [1, 2, 3]
#     circuit = CustomCircuit(circuit, initial_guess=initial_guess)
