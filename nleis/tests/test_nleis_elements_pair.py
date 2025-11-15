import string

import numpy as np

from nleis.nleis_elements_pair import (OverwriteError,  # noqa: F401
                                       circuit_elements, element, d,
                                       ElementError)  # noqa: F401
from impedance.models.circuits.circuits import CustomCircuit
from nleis.nleis import NLEISCustomCircuit  # noqa: F401

# supress warnings for cleaner test output
import warnings
warnings.filterwarnings("ignore")


def test_porous_electrode():

    # Test the convergence between
    # Porous electrode with high conductivity matrix model
    # with it corresponding TLM (charge transfer only)

    # EIS
    freqs = [0.001, 1.0, 1000, 100000]
    circuit_1 = CustomCircuit('TP', initial_guess=[1, 1, 1])
    circuit_2 = CustomCircuit('TLM', initial_guess=[1, 1, 1, 0, 0, 1000])
    Z1 = circuit_1.predict(freqs)
    Z2 = circuit_2.predict(freqs)
    assert np.allclose(Z1, Z2, atol=1e-3)

    # 2nd-NLEIS
    circuit_1 = NLEISCustomCircuit('TPn', initial_guess=[1, 1, 1, 0.1])
    circuit_2 = NLEISCustomCircuit('TLMn',
                                   initial_guess=[1, 1, 1, 0, 0, 1000, .1, 0])
    Z1 = circuit_1.predict(freqs, max_f=np.inf)
    Z2 = circuit_2.predict(freqs, max_f=np.inf)
    assert np.allclose(Z1, Z2, atol=1e-3)

    # Test the convergence between
    # Porous electrode with high conductivity matrix
    # and diffusion into spherical particles
    # with it corresponding TLM

    # EIS
    circuit_1 = CustomCircuit('TDS', initial_guess=[1, 1, 1, 1, 1])
    circuit_2 = CustomCircuit('TLMS',
                              initial_guess=[1, 1, 1, 1, 1, 0, 0, 1000])
    Z1 = circuit_1.predict(freqs)
    Z2 = circuit_2.predict(freqs)
    assert np.allclose(Z1, Z2, atol=1e-3)

    # 2nd-NLEIS
    circuit_1 = NLEISCustomCircuit('TDSn',
                                   initial_guess=[1, 1, 1, 1, 1, 1, 0.1])
    circuit_2 = NLEISCustomCircuit('TLMSn',
                                   initial_guess=[1, 1, 1, 1, 1, 0, 0, 1000,
                                                  1, .1, 0])
    Z1 = circuit_1.predict(freqs, max_f=np.inf)
    Z2 = circuit_2.predict(freqs, max_f=np.inf)
    assert np.allclose(Z1, Z2, atol=1e-3)

    # Test the convergence between
    # Porous electrode with high conductivity matrix
    # and planar diffusion into platelet-like particles
    # with it corresponding TLM
    circuit_1 = CustomCircuit('TDP', initial_guess=[1, 1, 1, 1, 1])
    circuit_2 = CustomCircuit('TLMD',
                              initial_guess=[1, 1, 1, 1, 1, 0, 0, 1000])
    Z1 = circuit_1.predict(freqs)
    Z2 = circuit_2.predict(freqs)
    assert np.allclose(Z1, Z2, atol=1e-3)

    # 2nd-NLEIS
    circuit_1 = NLEISCustomCircuit('TDPn',
                                   initial_guess=[1, 1, 1, 1, 1, 1, 0.1])
    circuit_2 = NLEISCustomCircuit('TLMDn',
                                   initial_guess=[1, 1, 1, 1, 1, 0, 0, 1000,
                                                  1, .1, 0])
    Z1 = circuit_1.predict(freqs, max_f=np.inf)
    Z2 = circuit_2.predict(freqs, max_f=np.inf)
    assert np.allclose(Z1, Z2, atol=1e-3)

    # TDC should converge to TP when Aw is zero
    circuit_1 = CustomCircuit('TP', initial_guess=[1, 1, 1])
    circuit_2 = CustomCircuit('TDC',
                              initial_guess=[1, 1, 1, 0, 10])
    Z1 = circuit_1.predict(freqs)
    Z2 = circuit_2.predict(freqs)
    assert np.allclose(Z1, Z2, atol=1e-3)

    # TDCn should converge to TPn when Aw and k is zero
    circuit_1 = CustomCircuit('TPn', initial_guess=[1, 1, 1, 0.1])
    circuit_2 = CustomCircuit('TDCn',
                              initial_guess=[1, 1, 1, 0, 10, 0, 0.1])
    Z1 = circuit_1.predict(freqs)
    Z2 = circuit_2.predict(freqs)
    assert np.allclose(Z1, Z2, atol=1e-3)

    # The second harmonic current distribution function should converge
    # to each other given the same charge transfer parameters while
    # ignoring the solid state diffusion
    circuit_1 = CustomCircuit('mTiDn',
                              initial_guess=[1, 1, 1, 0, 1, 1, 1,
                                             100, 0, .1, .1])
    circuit_2 = CustomCircuit('mTiSn',
                              initial_guess=[1, 1, 1, 0, 1, 1, 1,
                                             100, 0, .1, .1])
    Z1 = circuit_1.predict(freqs)
    Z2 = circuit_2.predict(freqs)
    assert np.allclose(Z1, Z2, atol=1e-3)


def test_TLM():
    freqs = [0.001, 1.0, 1000, 100000]

    # Test to make sure single and 2 element works 2nd-NLEIS
    circuit_1 = NLEISCustomCircuit('TLMn',
                                   initial_guess=[0, 1, 1, 0, 0, 1, .1, 0])
    circuit_2 = NLEISCustomCircuit('TLMn',
                                   initial_guess=[0, 1, 1, 0, 0, 2, .1, 0])

    Z1 = circuit_1.predict(freqs, max_f=np.inf)
    Z2 = circuit_2.predict(freqs, max_f=np.inf)
    assert np.allclose(Z1, Z2, atol=1e-3)

    circuit_1 = NLEISCustomCircuit('TLMSn',
                                   initial_guess=[0, 1, 1, 1, 1, 0, 0, 1,
                                                  1, .1, 0])
    circuit_2 = NLEISCustomCircuit('TLMSn',
                                   initial_guess=[0, 1, 1, 1, 1, 0, 0, 2,
                                                  1, .1, 0])
    Z1 = circuit_1.predict(freqs)
    Z2 = circuit_2.predict(freqs)
    assert np.allclose(Z1, Z2, atol=1e-3)

    circuit_1 = NLEISCustomCircuit('TLMDn',
                                   initial_guess=[0, 1, 1, 1, 1, 0, 0, 1,
                                                  1, .1, 0])
    circuit_2 = NLEISCustomCircuit('TLMDn',
                                   initial_guess=[0, 1, 1, 1, 1, 0, 0, 2,
                                                  1, .1, 0])
    Z1 = circuit_1.predict(freqs)
    Z2 = circuit_2.predict(freqs)
    assert np.allclose(Z1, Z2, atol=1e-3)

    # Test the current distribution function outputs

    circuit_1 = NLEISCustomCircuit('mTiSn',
                                   initial_guess=[0, 1, 1, 1, 1, 0, 0, 1,
                                                  1, .1, 0])
    circuit_2 = NLEISCustomCircuit('mTiSn',
                                   initial_guess=[0, 1, 1, 1, 1, 0, 0, 2,
                                                  1, .1, 0])
    Z1 = circuit_1.predict(1)
    Z2 = circuit_2.predict(1)

    # Test the current distribution function outputs
    assert np.allclose(np.sum(Z1), np.sum(Z2), atol=1e-3)
    circuit_1 = NLEISCustomCircuit('mTiDn',
                                   initial_guess=[0, 1, 1, 1, 1, 0, 0, 1,
                                                  1, .1, 0])
    circuit_2 = NLEISCustomCircuit('mTiDn',
                                   initial_guess=[0, 1, 1, 1, 1, 0, 0, 2,
                                                  1, .1, 0])
    Z1 = circuit_1.predict(1)
    Z2 = circuit_2.predict(1)
    assert np.allclose(np.sum(Z1), np.sum(Z2), atol=1e-3)


def test_RC():
    freqs = [0.001, 1.0, 1000]
    circuit_1 = CustomCircuit('RC', initial_guess=[1, 1])
    circuit_2 = CustomCircuit('p(R1,C1)', initial_guess=[1, 1])
    Z1 = circuit_1.predict(freqs)
    Z2 = circuit_2.predict(freqs)

    assert np.allclose(Z1, Z2)

    # Test the convergence between
    # Randles and Randles with CPE element (spherical diffusion)

    # EIS
    freqs = [0.001, 1.0, 1000]
    circuit_1 = CustomCircuit('RCS', initial_guess=[1, 1, 1, 1])
    circuit_2 = CustomCircuit('RCSQ', initial_guess=[1, 1, 1, 1, 1])
    Z1 = circuit_1.predict(freqs)
    Z2 = circuit_2.predict(freqs)
    assert np.allclose(Z1, Z2, atol=1e-3)

    # 2nd-NLEIS
    circuit_1 = NLEISCustomCircuit('RCSn', initial_guess=[1, 1, 1, 1, 1, 1])
    circuit_2 = NLEISCustomCircuit('RCSQn',
                                   initial_guess=[1, 1, 1, 1, 1, 1, 1])
    Z1 = circuit_1.predict(freqs)
    Z2 = circuit_2.predict(freqs)
    assert np.allclose(Z1, Z2, atol=1e-3)

    # Test the convergence between
    # Randles and Randles with CPE element (Planar diffusion)

    # EIS
    freqs = [0.001, 1.0, 1000]
    circuit_1 = CustomCircuit('RCD', initial_guess=[1, 1, 1, 1])
    circuit_2 = CustomCircuit('RCDQ', initial_guess=[1, 1, 1, 1, 1])
    Z1 = circuit_1.predict(freqs)
    Z2 = circuit_2.predict(freqs)
    assert np.allclose(Z1, Z2, atol=1e-3)

    # 2nd-NLEIS
    circuit_1 = NLEISCustomCircuit('RCDn', initial_guess=[1, 1, 1, 1, 1, 1])
    circuit_2 = NLEISCustomCircuit('RCDQn',
                                   initial_guess=[1, 1, 1, 1, 1, 1, 1])
    Z1 = circuit_1.predict(freqs)
    Z2 = circuit_2.predict(freqs)
    assert np.allclose(Z1, Z2, atol=1e-3)


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
