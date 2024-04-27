import string

import numpy as np

from nleis.nleis_elements_pair import (OverwriteError,
                                                circuit_elements, element, d,
                                                ElementError)
from impedance.models.circuits.circuits import CustomCircuit
from nleis.nleis import NLEISCustomCircuit

def test_porous_electrode():
   freqs = [0.001, 1.0, 1000]
   circuit_1 = CustomCircuit('TPO',initial_guess = [1,1,1])
   circuit_2 = CustomCircuit('TLM',initial_guess = [1,1,1,0,0,1000])
   Z1 = circuit_1.predict(freqs)
   Z2 = circuit_2.predict(freqs)
   assert np.allclose (Z1,Z2,atol=1e-3)



def test_RC():
   freqs = [0.001, 1.0, 1000]
   circuit_1 = CustomCircuit('RCO',initial_guess = [1,1])
   circuit_2 = CustomCircuit('p(R1,C1)',initial_guess = [1,1])
   Z1 = circuit_1.predict(freqs)
   Z2 = circuit_2.predict(freqs)

   assert np.allclose (Z1,Z2)



def test_d():
    a = np.array([5 + 6 * 1j, 2 + 3 * 1j])
    b = np.array([5 + 6 * 1j, 2 + 3 * 1j])

    answer = np.array([0, 0])
    assert np.isclose(d([a, b]), answer).all()



def test_element_function_names():
    # run a simple check to ensure there are no integers
    # in the function names
    letters = string.ascii_uppercase + string.ascii_lowercase

    for elem in circuit_elements.keys():
        print(elem)
        for char in elem:
            if elem.startswith('__'):
                continue
            assert (
                char in letters
            ), f"{char} in {elem} is not in the allowed set of {letters}"





