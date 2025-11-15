import numpy as np

import pytest

import os
from nleis.data_processing import data_loader, fft_phase_correction, \
    convert_to_complex, plot_fit, plot_freq_domain, data_truncation
import pandas as pd
# import matplotlib

# matplotlib.use("Agg")

test_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(test_dir, '../data')
data_path_25mA = os.path.join(data_dir, 'autolab_25mA.txt')
data_path_50mA = os.path.join(data_dir, 'autolab_50mA.txt')
data_path_100mA = os.path.join(data_dir, 'autolab_100mA.txt')

data_25mA_raw = data_loader(data_path_25mA, equipment='autolab',
                            fft='scipy', phase_correction=False,
                            baseline=False, multi_current=False)


def test_data_truncation():

    # # get example data
    # # The example is shown in "Getting Started" page

    frequencies = np.loadtxt(os.path.join(data_dir, 'freq_30a.txt'))
    Z1 = np.loadtxt(os.path.join(data_dir, 'Z1s_30a.txt')).view(complex)[1]
    Z2 = np.loadtxt(os.path.join(data_dir, 'Z2s_30a.txt')).view(complex)[1]

    max_f = 10

    f, Z1, Z2, f2_truncated, Z2_truncated = data_truncation(
        frequencies, Z1, Z2, max_f=max_f)
    assert len(f) == len(Z1) == len(Z2)
    assert Z1.imag.max() < 0
    assert f2_truncated.max() < max_f
    assert len(f2_truncated) == len(Z2_truncated)


def test_data_loader_multi_current():

    # test for multi_current
    # the data should be a list of data paths
    # otherwise raise ValueError
    with pytest.raises(ValueError):
        data_loader(data_path_25mA, equipment='autolab',
                    fft='instrument', multi_current=True)
    # test for multi_current
    # mismatched data path should lead to the same result
    path_list_1 = [data_path_25mA, data_path_50mA, data_path_100mA]
    path_list_2 = [data_path_100mA, data_path_25mA, data_path_50mA]
    result_list_1 = data_loader(path_list_1, equipment='autolab',
                                fft='instrument', multi_current=True)
    result_list_2 = data_loader(path_list_2, equipment='autolab',
                                fft='instrument', multi_current=True)
    assert np.allclose(
        result_list_1[1], result_list_2[1], rtol=1e-2, atol=1e-2)

    # # test to ensure fit_visualize can run without problem
    # data_loader(path_list_2, equipment='autolab',
    #             fft='instrument', multi_current=True,
    #             fit_visualize=True)


def test_data_loader_single_current():
    # test for single current
    # the data should be a single data path in string
    # otherwise raise ValueError
    with pytest.raises(ValueError):

        data_loader([data_path_25mA], equipment='autolab',
                    fft='instrument', multi_current=False)

    # test for phase correction
    # the phase corrected impedance
    # should match non-phase corrected impedance
    data_25mA_corrected = data_loader(data_path_25mA, equipment='autolab',
                                      fft='scipy', phase_correction=True,
                                      baseline=False, multi_current=False)
    assert np.allclose(
        data_25mA_raw[1], data_25mA_corrected[1], rtol=1e-2, atol=1e-2)

    # # test to ensure freq_domain_visual can run without problem
    # data_loader(data_path_25mA, equipment='autolab',
    #             fft='scipy', phase_correction=True,
    #             baseline=False, multi_current=False,
    #             freq_domain_visual=True)


def test_data_loader_baseline():
    # test for baseline correction
    # the baseline corrected impedance
    # should match non-baseline corrected impedance for EIS data

    data_25mA_baslined = data_loader(data_path_25mA, equipment='autolab',
                                     fft='scipy', phase_correction=True,
                                     baseline=True, multi_current=False)
    assert np.allclose(
        data_25mA_raw[1], data_25mA_baslined[1], rtol=1e-2, atol=1e-2)


def test_plot_fit():
    # test for plot_fit

    I1 = np.array([1, 2, 3, 4, 5])*(1+1*1j)
    V1 = np.array([1, 2, 3, 4, 5])*(2+2*1j)
    V2 = np.array([1, 2, 3, 4, 5])*(3+3*1j)
    V1_cal = np.array([1, 2, 3, 4, 5])*(1+1.1*1j)
    V2_cal = np.array([1, 2, 3, 4, 5])*(2+2.1*1j)
    scaling_factor = 1
    _, ax = plot_fit(I1, V1, V2, V1_cal, V2_cal, scaling_factor)
    # Check the x-data and y-data of each line
    x0, y0 = ax[0].lines[0].get_xydata().T
    assert (x0 == I1.real).all() and (y0 == V1.real).all()
    x1, y1 = ax[0].lines[1].get_xydata().T
    assert (x1 == I1.real).all() and (y1 == V2.real*scaling_factor).all()
    x2, y2 = ax[0].lines[2].get_xydata().T
    assert (x2 == I1.real).all() and (y2 == V1_cal.real).all()
    x3, y3 = ax[0].lines[3].get_xydata().T
    assert (x3 == I1.real).all() and (y3 == V2_cal.real*scaling_factor).all()
    x4, y4 = ax[1].lines[0].get_xydata().T
    assert (x4 == I1.real).all() and (y4 == V1.imag).all()
    x5, y5 = ax[1].lines[1].get_xydata().T
    assert (x5 == I1.real).all() and (y5 == V2.imag*scaling_factor).all()
    x6, y6 = ax[1].lines[2].get_xydata().T
    assert (x6 == I1.real).all() and (y6 == V1_cal.imag).all()
    x7, y7 = ax[1].lines[3].get_xydata().T
    assert (x7 == I1.real).all() and (y7 == V2_cal.imag*scaling_factor).all()


def test_plot_freq_domain():
    f = 1
    freq = np.array([0, 1, 2, 3, 4, 5])
    If = np.array([0, 1, 2, 3, 4, 5])*(1+1j)
    V = np.array([0, 1, 2, 3, 4, 5])*(1+1j)
    k = [1, 2, 3]
    N = len(k)
    _, ax = plot_freq_domain(freq, f, If, V, k)
    # Check the x-data and y-data of each line
    # for current and voltage
    x0, y0 = ax[0].lines[0].get_xydata().T

    assert np.allclose(x0, freq) and np.allclose(y0, abs(If))
    # (x0 == freq).all() and (y0 == abs(If)).all()
    x1, y1 = ax[1].lines[0].get_xydata().T
    assert np.allclose(x1, freq) and np.allclose(y1, abs(V))

    # Check the x-data and y-data of current harmonics
    x2, y2 = ax[0].lines[1].get_xydata().T
    assert np.allclose(x2, freq[k[0]]) and np.allclose(y2, abs(If[k[0]]))

    # Check the x-data and y-data of voltage harmonics
    for i in range(N):
        x3, y3 = ax[1].lines[i+1].get_xydata().T
        assert np.allclose(x3, freq[k[i]]) and np.allclose(y3, abs(If[k[i]]))

    # test if the input data is a pandas dataframe
    df_freq = pd.DataFrame(freq)
    df_If = pd.DataFrame(If)
    df_V = pd.DataFrame(V)

    _, ax = plot_freq_domain(df_freq, f, df_If, df_V, k)
    # Repeat the same test
    # Check the x-data and y-data of each line
    # for current and voltage
    x0, y0 = ax[0].lines[0].get_xydata().T
    assert np.allclose(x0, freq) and np.allclose(y0, abs(If))
    x1, y1 = ax[1].lines[0].get_xydata().T
    assert np.allclose(x1, freq) and np.allclose(y1, abs(V))

    # Check the x-data and y-data of current harmonics
    x2, y2 = ax[0].lines[1].get_xydata().T
    assert np.allclose(x2, freq[k[0]]) and np.allclose(y2, abs(If[k[0]]))
    # Check the x-data and y-data of voltage harmonics
    for i in range(N):
        x3, y3 = ax[1].lines[i+1].get_xydata().T
        assert np.allclose(x3, freq[k[i]]) and np.allclose(y3, abs(If[k[i]]))


def test_fft_phase_correction():
    # test for fft_phase_correction
    # return ValueError when number of cycles
    # is greater than length of data
    current = [1, 2]
    voltage = [1, 2]
    num_cycles = 3
    with pytest.raises(ValueError):
        fft_phase_correction(current, voltage, num_cycles)


def test_convert_to_complex():
    # test for convert_to_complex
    # return 0 when input is nan
    asc_str = float('nan')
    assert np.isclose(convert_to_complex(asc_str), 0)
