import numpy as np
import pytest

from nleis import parallel
from nleis import EISandNLEIS
from impedance.models.circuits import CustomCircuit

# supress warnings for cleaner test output
import warnings
warnings.filterwarnings("ignore")

circuit_1 = 'RC0'
circuit_2 = 'RCn0'
circuit_eis = 'p(R,C)'
initial_guess_1 = [1.0, 0.1, 0.5]
initial_guess_2 = [1.0, 0.1]
model = EISandNLEIS(circuit_1, circuit_2, initial_guess_1)
f = np.geomspace(1e-3, 1e3, 60)
Z1, Z2 = model.predict(f)


def test_fit_once(capsys):
    # Test to make sure the backward compatibility of impedance.py
    eis_model = CustomCircuit(circuit_eis, initial_guess=initial_guess_2)

    result = parallel.fit_once(
        eis_model,
        initial_guess=initial_guess_2,
        f=f,
        show_results=True,
        impedance=Z1,
    )

    assert result["Status"] is True
    assert "Status" in capsys.readouterr().out

    # Test to make sure failed attempts are returned as well
    result = parallel.fit_once(
        eis_model,
        initial_guess=0,
        f=f,
        show_results=True,
        impedance=Z1,
    )
    assert result["Status"] is False
    assert "Status" in capsys.readouterr().out

    # Test to make sure exception in the input data is handled
    result = parallel.fit_once(
        eis_model,
        initial_guess=0,
        f=f,
        show_results=True,
    )
    assert result["err"] in capsys.readouterr().out

    # Test to make sure the fitting works correctly for simultaneous fitting
    # of EIS and 2nd-NLEIS
    model = EISandNLEIS(circuit_1, circuit_2, initial_guess_1)
    result = parallel.fit_once(
        model,
        initial_guess=initial_guess_1,
        f=f,
        show_results=True,
        Z1=Z1,
        Z2=Z2,
    )
    assert result["Status"] is True
    assert "Status" in capsys.readouterr().out


def test_multistart_fit(capsys):
    # Again test to ensure the backward compatibility of impedance.py
    eis_model = CustomCircuit(circuit_eis, initial_guess=initial_guess_2)
    best, results = parallel.multistart_fit(
        eis_model,
        f=f,
        sampling_method="sobol",
        num_samples=2,
        show_results=True,
        impedance=Z1,
    )
    assert best["Status"] is True
    assert len(results) == 13  # 12 from sobol sampling + 1 initial guess

    # Test to make sure random sampling works
    best, results = parallel.multistart_fit(
        eis_model,
        f=f,
        sampling_method="random",
        num_samples=2,
        show_results=True,
        impedance=Z1,
    )
    assert best["Status"] is True
    assert len(results) == 3  # 2 from random sampling + 1 initial guess

    # Test to make sure custom sampling works
    initial_guesses = [[1, 2], [3, 4]]
    best, results = parallel.multistart_fit(
        eis_model,
        f=f,
        sampling_method="custom",
        num_samples=2,
        initial_guesses=initial_guesses,
        show_results=True,
        impedance=Z1,
    )
    assert best["Status"] is True
    assert len(results) == 3  # 2 from custom sampling + 1 initial guess

    # Test to make sure ValueError is raised for when no initial guesses are
    # provided for custom sampling
    with pytest.raises(ValueError):
        best, results = parallel.multistart_fit(
            eis_model,
            f=f,
            sampling_method="custom",
            num_samples=2,
            show_results=True,
            impedance=Z1,
        )
    # Test to make sure Unsupported sampling method raises ValueError
    with pytest.raises(ValueError):
        best, results = parallel.multistart_fit(
            eis_model,
            f=f,
            sampling_method="unsupported_method",
            num_samples=2,
            show_results=True,
            impedance=Z1,
        )

    # Test to make sure RuntimeError is raised when all fits fail
    with pytest.raises(RuntimeError):
        best, results = parallel.multistart_fit(
            eis_model,
            f=f,
            sampling_method="sobol",
            num_samples=2,
            show_results=True,
            impedance=None,
        )
