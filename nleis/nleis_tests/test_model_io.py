import numpy as np
from nleis.nleis import EISandNLEIS
import os

import os.path
import sys


def test_model_io():

    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(test_dir, '../data')

    # # get example data
    # # The example is shown in "Getting Started" page 
    
    frequencies = np.loadtxt(os.path.join(data_dir, 'freq_30a.txt'))
    Z1 = np.loadtxt(os.path.join(data_dir,'Z1s_30a.txt')).view(complex)[1]
    Z2 = np.loadtxt(os.path.join(data_dir,'Z2s_30a.txt')).view(complex)[1]
    circ_str_1 = 'L0-R0-TDS0-TDS1'
    circ_str_2 = 'd(TDSn0,TDSn1)'

    initial_guess = [1e-7,1e-3 # L0,RO
                    ,5e-3,1e-3,10,1e-2,100,10,0.1 ## TDS0 + additioal nonlinear parameters
                    ,1e-3,1e-3,1e-3,1e-2,1000,0,0 ## TDS1 + additioal nonlinear parameters
                    ]

    circuit_1 = EISandNLEIS(circ_str_1,circ_str_2,initial_guess=initial_guess)

    # circuit_1.save('./nleis_tests/test_io.json')
    circuit_1.save(os.path.join(data_dir,'test_io.json'))

    circuit_2 = EISandNLEIS()
    circuit_2.load(os.path.join(data_dir,'test_io.json'))

    assert circuit_1 == circuit_2

    circuit_1.fit(frequencies, Z1,Z2)
    p_fit = list(circuit_1.parameters_)
    circuit_1.save(os.path.join(data_dir,'test_io.json'))
    circuit_2 = EISandNLEIS()
    circuit_2.load(os.path.join(data_dir,'test_io.json'))

    assert str(circuit_1) == str(circuit_2)
    assert circuit_1 == circuit_2

    fitted_template = EISandNLEIS()
    fitted_template.load(os.path.join(data_dir,'test_io.json'), fitted_as_initial=True)
    circuit_1 = EISandNLEIS(circ_str_1,circ_str_2,initial_guess=p_fit)
    assert str(circuit_1) == str(fitted_template)
    assert circuit_1 == fitted_template
