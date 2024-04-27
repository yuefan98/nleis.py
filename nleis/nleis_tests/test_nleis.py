import json
import os

import numpy as np
import matplotlib.pyplot as plt
import pytest

from impedance.models.circuits import BaseCircuit, CustomCircuit, Randles
from nleis.nleis import EISandNLEIS, NLEISCustomCircuit

test_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(test_dir, '../data')

# # get example data
# # The example is shown in "Getting Started" page 

frequencies = np.loadtxt(os.path.join(data_dir, 'freq_30a.txt'))
Z1 = np.loadtxt(os.path.join(data_dir,'Z1s_30a.txt')).view(complex)[1]
Z2 = np.loadtxt(os.path.join(data_dir,'Z2s_30a.txt')).view(complex)[1]




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

    # check _is_fit()
    assert not NLEIS_circuit._is_fit()

    # check complex frequencies raise TypeError
    with pytest.raises(TypeError):
        NLEIS_circuit.predict([0.42, 42 + 42j])


