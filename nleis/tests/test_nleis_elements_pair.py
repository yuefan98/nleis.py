import string

import numpy as np

import nleis.nleis_elements_pair as elements
from nleis.nleis_elements_pair import (OverwriteError,  # noqa: F401
                                       circuit_elements, element, d,
                                       ElementError)  # noqa: F401
from impedance.models.circuits.circuits import CustomCircuit
from nleis.nleis import NLEISCustomCircuit  # noqa: F401

# supress warnings for cleaner test output
import warnings
warnings.filterwarnings("ignore")


def dense_linear_current_distribution(Zran, Rpore, N):
    Req = np.copy(Zran)
    inv_Zran = 1 / Zran
    for _ in range(1, N):
        Req = 1 / ((1 / (Req + Rpore)) + inv_Zran)
    Req = Req + Rpore

    I1 = np.zeros((len(Zran), N), dtype=np.complex128)
    for freq_idx in range(len(Zran)):
        Ax = np.eye(N, dtype=np.complex128) * Zran[freq_idx]
        i_idx, j_idx = np.tril_indices(N, -1)
        Ax[i_idx, j_idx] = -(i_idx - j_idx) * Rpore
        Ax += np.arange(1, N + 1).reshape(-1, 1) * Rpore
        b = np.ones(N, dtype=np.complex128) * Req[freq_idx]
        I1[freq_idx, :] = np.linalg.solve(Ax, b)
    return I1


def dense_second_harmonic_current(N, Rpore, Z12t, rhs):
    Ax = elements.A_matrices_TLMn(N, Rpore, Z12t)
    return np.linalg.solve(Ax, rhs)


def dense_second_harmonic_impedance(I1, Rpore, Z2, Z12t):
    Z = np.zeros(len(Z2), dtype=np.complex128)
    N = I1.shape[1]
    for freq_idx in range(len(Z2)):
        Ii = I1[freq_idx]
        b = np.zeros(N, dtype=np.complex128)
        for i in range(N - 1):
            b[i] = Ii[-1]**2 - Ii[i]**2
        I2 = dense_second_harmonic_current(
            N, Rpore, Z12t[freq_idx], -b * Z2[freq_idx]
        )
        Z[freq_idx] = Z2[freq_idx] * Ii[0]**2 + I2[-1] * Z12t[freq_idx]
    return Z


def dense_second_harmonic_distribution(I1, Rpore, Z2, Z12t):
    N = I1.shape[1]
    I2 = np.zeros_like(I1)
    for freq_idx in range(len(Z2)):
        Ii = I1[freq_idx]
        b = np.zeros(N, dtype=np.complex128)
        for i in range(N - 1):
            b[i] = Ii[-1]**2 - Ii[i]**2
        I2[freq_idx, :] = dense_second_harmonic_current(
            N, Rpore, Z12t[freq_idx], -b * Z2[freq_idx]
        )[::-1]
    return I2


def test_tlm_linear_current_distribution_matches_dense_reference():
    freqs = np.logspace(-3, 4, 9)

    N = 18
    p = [1.7, 0.8, 0.6, 1.2, 0.4, N]
    Rct, Cdl = p[1] * N, p[2] / N
    Rs, Cs = p[3] * N, p[4] / N
    Zran = elements.RC([Rct, Cdl], freqs) + elements.RC([Rs, Cs], freqs)
    expected = dense_linear_current_distribution(Zran, p[0] / N, N)
    assert np.allclose(elements.mTi(p, freqs), expected, rtol=1e-11)

    p = [1.7, 0.8, 0.6, 0.9, 2.0, 1.2, 0.4, N]
    Rct, Cdl = p[1] * N, p[2] / N
    Aw, τd = p[3] * N, p[4]
    Rs, Cs = p[5] * N, p[6] / N
    Zran = elements.RCS([Rct, Cdl, Aw, τd], freqs) + \
        elements.RC([Rs, Cs], freqs)
    expected = dense_linear_current_distribution(Zran, p[0] / N, N)
    assert np.allclose(elements.mTiS(p, freqs), expected, rtol=1e-11)

    Zran = elements.RCD([Rct, Cdl, Aw, τd], freqs) + \
        elements.RC([Rs, Cs], freqs)
    expected = dense_linear_current_distribution(Zran, p[0] / N, N)
    assert np.allclose(elements.mTiD(p, freqs), expected, rtol=1e-11)


def test_tlm_second_harmonic_solver_matches_dense_reference():
    rng = np.random.default_rng(10)
    for N in [2, 5, 25, 100]:
        for _ in range(3):
            Rpore = 0.1 + rng.random()
            Z12t = 0.2 + rng.random() + 1j * (0.1 + rng.random())
            rhs = rng.normal(size=N) + 1j * rng.normal(size=N)
            expected = dense_second_harmonic_current(N, Rpore, Z12t, rhs)
            actual = elements.solve_tlm_second_harmonic_current(
                N, Rpore, Z12t, rhs
            )
            assert np.allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_tlm_second_harmonic_elements_match_dense_reference():
    freqs = np.logspace(-3, 4, 9)

    N = 20
    p = [1.7, 0.8, 0.6, 1.2, 0.4, N, 0.3, -0.2]
    Rpore = p[0] / N
    Rct, Cdl = p[1] * N, p[2] / N
    Rs, Cs = p[3] * N, p[4] / N
    Z1 = elements.RC([Rct, Cdl], freqs) + elements.RC([Rs, Cs], freqs)
    I1 = dense_linear_current_distribution(Z1, Rpore, N)
    Z2 = elements.RCn([Rct, Cdl, p[6]], freqs) + \
        elements.RCn([Rs, Cs, p[7]], freqs)
    Z12t = elements.RC([Rct, Cdl], 2 * freqs) + \
        elements.RC([Rs, Cs], 2 * freqs)
    expected = dense_second_harmonic_impedance(I1, Rpore, Z2, Z12t)
    assert np.allclose(elements.TLMn(p, freqs), expected, rtol=1e-10)

    p = [1.7, 0.8, 0.6, 0.9, 2.0, 1.2, 0.4, N, 0.2, 0.3, -0.2]
    Rpore = p[0] / N
    Rct, Cdl = p[1] * N, p[2] / N
    Aw, τd = p[3] * N, p[4]
    Rs, Cs = p[5] * N, p[6] / N
    Z1 = elements.RCS([Rct, Cdl, Aw, τd], freqs) + \
        elements.RC([Rs, Cs], freqs)
    I1 = dense_linear_current_distribution(Z1, Rpore, N)
    Z2 = elements.RCSn([Rct, Cdl, Aw, τd, p[8], p[9]], freqs) + \
        elements.RCn([Rs, Cs, p[10]], freqs)
    Z12t = elements.RCS([Rct, Cdl, Aw, τd], 2 * freqs) + \
        elements.RC([Rs, Cs], 2 * freqs)
    expected = dense_second_harmonic_impedance(I1, Rpore, Z2, Z12t)
    assert np.allclose(elements.TLMSn(p, freqs), expected, rtol=1e-10)

    Z1 = elements.RCD([Rct, Cdl, Aw, τd], freqs) + \
        elements.RC([Rs, Cs], freqs)
    I1 = dense_linear_current_distribution(Z1, Rpore, N)
    Z2 = elements.RCDn([Rct, Cdl, Aw, τd, p[8], p[9]], freqs) + \
        elements.RCn([Rs, Cs, p[10]], freqs)
    Z12t = elements.RCD([Rct, Cdl, Aw, τd], 2 * freqs) + \
        elements.RC([Rs, Cs], 2 * freqs)
    expected = dense_second_harmonic_impedance(I1, Rpore, Z2, Z12t)
    assert np.allclose(elements.TLMDn(p, freqs), expected, rtol=1e-10)


def test_tlm_second_harmonic_current_outputs_match_dense_reference():
    freqs = np.logspace(-3, 4, 9)
    N = 20
    p = [1.7, 0.8, 0.6, 0.9, 2.0, 1.2, 0.4, N, 0.2, 0.3, -0.2]

    Rpore = p[0] / N
    Rct, Cdl = p[1] * N, p[2] / N
    Aw, τd = p[3] * N, p[4]
    Rs, Cs = p[5] * N, p[6] / N

    Z1 = elements.RCS([Rct, Cdl, Aw, τd], freqs) + \
        elements.RC([Rs, Cs], freqs)
    I1 = dense_linear_current_distribution(Z1, Rpore, N)
    Z2 = elements.RCSn([Rct, Cdl, Aw, τd, p[8], p[9]], freqs) + \
        elements.RCn([Rs, Cs, p[10]], freqs)
    Z12t = elements.RCS([Rct, Cdl, Aw, τd], 2 * freqs) + \
        elements.RC([Rs, Cs], 2 * freqs)
    expected = dense_second_harmonic_distribution(I1, Rpore, Z2, Z12t)
    assert np.allclose(elements.mTiSn(p, freqs), expected, rtol=1e-10)

    Z1 = elements.RCD([Rct, Cdl, Aw, τd], freqs) + \
        elements.RC([Rs, Cs], freqs)
    I1 = dense_linear_current_distribution(Z1, Rpore, N)
    Z2 = elements.RCDn([Rct, Cdl, Aw, τd, p[8], p[9]], freqs) + \
        elements.RCn([Rs, Cs, p[10]], freqs)
    Z12t = elements.RCD([Rct, Cdl, Aw, τd], 2 * freqs) + \
        elements.RC([Rs, Cs], 2 * freqs)
    expected = dense_second_harmonic_distribution(I1, Rpore, Z2, Z12t)
    assert np.allclose(elements.mTiDn(p, freqs), expected, rtol=1e-10)


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
