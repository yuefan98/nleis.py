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

    result = parallel.fit_once(
        eis_model,
        initial_guess=[1, 1],
        f=f,
        show_results=True,
        impedance=Z1,
        Z1=Z1,
        Z2=Z2,
    )
    assert result["Status"] is False
    assert "ValueError" in capsys.readouterr().out

    # Test to make sure exception in the input data is handled
    result = parallel.fit_once(
        eis_model,
        initial_guess=[1, 1],
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


def test_multistart_fit():
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


def test_batch_data_fit():

    # Again test to ensure the backward compatibility of impedance.py
    eis_model = CustomCircuit(circuit_eis, initial_guess=initial_guess_2)
    results = parallel.batch_data_fit(
        eis_model,
        f=f,
        impedance_list=[Z1, Z1*2],
    )
    assert results[0]["Status"] is True

    # Test to make sure exception in the input data is handled
    # and value error message is printed if no impedance data is provided
    with pytest.raises(ValueError):
        results = parallel.batch_data_fit(
            eis_model,
            f=f,
            show_results=True,
        )
    # Test to make sure exception in the input data is handled
    # and value error message is printed if too many impedance data is provided
    with pytest.raises(ValueError):
        results = parallel.batch_data_fit(
            eis_model,
            impedance_list=[Z1, Z2],
            Z1_list=[Z1, Z2],
            Z2_list=[Z1, Z2],
            f=f,
            show_results=True,
        )

    # Test to make sure RuntimeError is raised when all fits fail
    with pytest.raises(RuntimeError):
        results = parallel.batch_data_fit(
            eis_model,
            f=f,
            impedance_list=[np.inf, np.inf],
        )
    # Test to make sure the fitting works correctly for simultaneous fitting
    # of EIS and 2nd-NLEIS
    model = EISandNLEIS(circuit_1, circuit_2, initial_guess_1)
    results = parallel.batch_data_fit(
        model,
        f=f,
        Z1_list=[Z1, Z1],
        Z2_list=[Z2, Z2],
        show_results=True,
    )
    # print(result)
    assert results[0]["Status"] is True


def test_batch_model_fit():

    # Again test to ensure the backward compatibility of impedance.py
    eis_model_1 = CustomCircuit(circuit_eis, initial_guess=initial_guess_2)
    eis_model_2 = CustomCircuit(circuit_eis, initial_guess=initial_guess_2)
    results = parallel.batch_model_fit(
        [eis_model_1, eis_model_2],
        f=f,
        impedance=Z1,
    )
    assert results[0]["Status"] is True

    # Test to make sure RuntimeError is raised when all fits fail
    with pytest.raises(RuntimeError):
        results = parallel.batch_model_fit(
            [eis_model_1, eis_model_2],
            f=f,
        )
