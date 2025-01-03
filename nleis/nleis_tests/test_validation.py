import matplotlib.pyplot as plt
import numpy as np
from nleis.validation import MM, cost_max_norm
from impedance.models.circuits import CustomCircuit
from nleis import NLEISCustomCircuit
import pytest


# Suppress plt.show()
@pytest.fixture(autouse=True)
def suppress_plot_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)


def test_MM_cost_method():

    model = NLEISCustomCircuit('RCn', initial_guess=[.1, 1, 0.1])
    f = np.geomspace(1e-2, 1e2, 100)
    Z2 = model.predict(f, max_f=np.inf)

    # test invalid max_M raises value error
    with pytest.raises(ValueError):
        M, cost, p, Z_fit, res_real, res_imag, conf = MM(
            f, Z2, raw_circuit='Kn', initial_guess=[.1, 1], method='cost',
            max_f=10, max_M=0, tol=1e-5, plot=True)

    # check for an ideal weakly nonlinear RC, M equal to 1
    M, cost, p, Z_fit, res_real, res_imag, conf = MM(
        f, Z2, raw_circuit='Kn', initial_guess=[.1, 1], method='cost',
        max_f=10, max_M=10, tol=1e-5, plot=True)
    assert np.isclose(M, 1)

    model = CustomCircuit('TPn0', initial_guess=[1, 1, 1, 0.1])
    Z2 = model.predict(f)

    # test optimal solution unfound with max_M raises value error
    with pytest.raises(ValueError):
        M, cost, p, Z_fit, res_real, res_imag, conf = MM(
            f, Z2, raw_circuit='Kn', initial_guess=[.1, 1], method='cost',
            max_f=10, max_M=2, tol=1e-5, plot=True)


def test_MM_conf_method():
    model = CustomCircuit('p(R0,C0)', initial_guess=[.1, 1])
    f = np.geomspace(1e-2, 1e2, 100)
    Z1 = model.predict(f)

    # test invalid max_M raises value error
    with pytest.raises(ValueError):
        M, p, Z_fit, res_real, res_imag, conf = MM(
            f, Z1, raw_circuit='K', initial_guess=[.1, .1], method='conf',
            max_M=0, plot=True, CI_plot=True)

    # check for an ideal RC, M equal to 1
    M, p, Z_fit, res_real, res_imag, conf = MM(
        f, Z1, raw_circuit='K', initial_guess=[.1, .1], method='conf',
        max_M=10, plot=True, CI_plot=True)
    print(M)
    assert np.isclose(M, 1)

    model = CustomCircuit('TP0', initial_guess=[1, 1, 1])
    f = np.geomspace(1e-2, 1e2, 100)
    Z1 = model.predict(f)

    # test optimal solution unfound with max_M raises value error
    with pytest.raises(ValueError):
        M, p, Z_fit, res_real, res_imag, conf = MM(
            f, Z1, raw_circuit='K', initial_guess=[.1, .1], method='conf',
            max_M=3, plot=True, CI_plot=True)

    # test faliure of conf method raises value error
    with pytest.raises(ValueError):
        M, p, Z_fit, res_real, res_imag, conf = MM(
            f, -Z1, raw_circuit='K', initial_guess=[.1, .1], method='conf',
            max_M=3, plot=True, CI_plot=True)


def test_MM_method():
    # test invalid method raises value error
    with pytest.raises(ValueError):
        MM(np.array([1, 2, 3]), np.array([1, 2, 3]), raw_circuit='K',
           initial_guess=[1, 1], method='invalid_method')
    # test non-integer max_M raises TypeError for both method
    with pytest.raises(TypeError):
        MM(np.array([1, 2, 3]), np.array([1, 2, 3]), raw_circuit='K',
           initial_guess=[1, 1], method='cost', max_M=1.5)
    with pytest.raises(TypeError):
        MM(np.array([1, 2, 3]), np.array([1, 2, 3]), raw_circuit='K',
           initial_guess=[1, 1], method='conf', max_M=1.5)


def test_cost_max_norm():
    data = np.array([1 + 1j, 2 + 3*1j])
    model = np.array([1 + 1.2 * 1j, 2 + 2.9*1j])

    cost = cost_max_norm(data, model)

    assert np.isclose(cost, 0.00384615384615385)
