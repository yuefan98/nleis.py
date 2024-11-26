import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def data_loader(filename, equipment='autolab', fft='instrument',
                multi_current=False, rtol=5e-4, scaling_factor=1000,
                fit_visualize=False, freq_domain_visual=False):
    """
    Load data and perform FFT processing for EIS and 2nd-NLEIS data.

    Parameters:
    -----------
    filename : str or list of str
        Path(s) to the CSV file(s) containing the raw data.
        list of str is required if using multi_current mode.
    equipment : str, default 'autolab'
        Specifies the equipment used for data acquisition.
    fft : str, default 'instrument'
        Method used for FFT. Can be 'scipy' or 'instrument'.
    multi_current : bool, default False
        If True, process multiple current files.
    rtol : float, default 5e-4
        Relative tolerance for finding the frequency components.
    scaling_factor : int, default 1000
        Scaling factor for visualizing 2nd-NLEIS results.
    fit_visualize : bool, default False
        If True, generate plots to visualize the impedance fit
        to multiple current.
        Only available under multi_current mode
    freq_domain_visual: bool, default False
        If True, generate frequency domain current and voltage responses
        at each frequency for the first three harmonics.
        Only available when multi_current = False


    Returns:
    --------
    frequencies : ndarray
        Array of frequency values.
    Z1 : ndarray
        Impedance values for EIS.
    Z2 : ndarray
        Impedance values for 2nd-NLEIS.
    """
    if equipment == 'autolab':
        if multi_current:
            if not isinstance(filename, list):
                raise ValueError(
                    "filename should be a list of" +
                    " file paths when multi_current=True")

            df_I1, df_V1, df_V2 = pd.DataFrame(), pd.DataFrame(), \
                pd.DataFrame()

            for index, file in enumerate(filename):
                df = pd.read_csv(file)
                results = autolab_fft(df, method=fft, rtol=rtol)
                df_I1[index] = results['I1,[A]']
                df_V1[index] = results['V1,[V]']
                df_V2[index] = results['V2,[V]']

            frequencies = df['Frequency (Hz)'].dropna()
            Z1, Z2 = [], []

            for i in range(len(frequencies)):
                I1 = df_I1.iloc[i].to_numpy().astype(np.complex128)
                V1 = df_V1.iloc[i].to_numpy().astype(np.complex128)
                V2 = df_V2.iloc[i].to_numpy().astype(np.complex128)

                Z1_popt, _ = curve_fit(fit_Z1, np.vstack(
                    [I1.real, I1.imag]), np.hstack([V1.real, V1.imag]))
                Z2_popt, _ = curve_fit(fit_Z2, np.vstack(
                    [I1.real, I1.imag]), np.hstack([V2.real, V2.imag]))

                Z1.append(Z1_popt[0] + 1j * Z1_popt[1])
                Z2.append(Z2_popt[0] + 1j * Z2_popt[1])

                V1_cal, V2_cal = I1 * \
                    Z1[-1], (I1 ** 2) * Z2[-1] * scaling_factor

                if fit_visualize:
                    fig, ax = plot_fit(I1, V1, V2, V1_cal,
                                       V2_cal, scaling_factor)

            return frequencies.to_numpy(), np.array(Z1), np.array(Z2)

        else:
            if not isinstance(filename, str):
                raise ValueError(
                    "filename should be a string file " +
                    "paths when multi_current=False")
            df = pd.read_csv(filename)
            results = autolab_fft(df, method=fft, rtol=rtol,
                                  freq_domain_visual=freq_domain_visual)
            Z1 = results['V1,[V]'] / results['I1,[A]']
            Z2 = results['V2,[V]'] / (results['I1,[A]'] ** 2)

            return (results['freq,[Hz]'].to_numpy(),
                    Z1.to_numpy().astype(np.complex128),
                    Z2.to_numpy().astype(np.complex128))


def autolab_fft(df, method='scipy', rtol=5e-4, max_k=3,
                freq_domain_visual=False):
    """
    Perform FFT on the Autolab data and
    extract the relevant frequency components.

    Parameters:
    -----------
    df : DataFrame
        Input data from Autolab.
    method : str, default 'scipy'
        FFT method to use. Can be 'scipy' or 'instrument'.
    rtol : float, default 5e-4
        Relative tolerance for matching frequency components.

    Returns:
    --------
    DataFrame
        A DataFrame containing the frequencies
        and the calculated current/voltage values.
    """
    frequencies = df['Frequency (Hz)'].dropna()
    loc = 0
    results = pd.DataFrame(columns=[
                           'freq,[Hz]']+[f'I{i+1},[A]' for i in range(max_k)]
                           + [f'V{i+1},[V]' for i in range(max_k)])
    results['freq,[Hz]'] = frequencies

    if method == 'scipy':
        index = df[df['Time domain (s)'] == 0].index[1]

        for i, f in enumerate(frequencies):
            data = df.iloc[loc:loc+index]
            I, V = fft_data(data, rtol, f, max_k=max_k,
                            freq_domain_visual=freq_domain_visual)
            for j in range(max_k):
                results.loc[i, f'I{j+1},[A]'] = I['I'+str(j+1)]
                results.loc[i, f'V{j+1},[V]'] = V['V'+str(j+1)]

            loc += index

    elif method == 'instrument':

        index = df[df['Frequency domain (Hz)'] == 0].index[1]
        # extract instrument fft results
        for i, f in enumerate(frequencies):
            data = df.iloc[loc:loc+index]
            freq = data['Frequency domain (Hz)'].astype('float64')
            If = data['Current frequency domain'].apply(
                convert_to_complex).astype('complex128')
            V = data['Potential frequency domain'].apply(
                convert_to_complex).astype('complex128')
            I_dict = {}
            V_dict = {}
            k = []
            for j in range(1, max_k+1):

                k_j = data[np.isclose(
                    data['Frequency domain (Hz)'], j*f)].index-i*index
                k.append(k_j)
                I_dict['I'+str(j)] = If.iloc[k_j].to_numpy()
                V_dict['V'+str(j)] = V.iloc[k_j].to_numpy()

            for j in range(max_k):
                results.loc[i, f'I{j+1},[A]'] = I_dict['I'+str(j+1)]
                results.loc[i, f'V{j+1},[V]'] = V_dict['V'+str(j+1)]

            if freq_domain_visual:
                plot_freq_domain(freq, f, If, V, k)

            loc += index

    return results


def fft_data(data, rtol, f, max_k=3, freq_domain_visual=False):
    """Perform FFT for each frequency
    and return the current and voltage values."""
    N = len(data)
    sampling_rate = data['Time domain (s)'].iloc[1]
    freq = rfftfreq(N, sampling_rate)

    If = rfft(data['Current (AC) (A)'].to_numpy()) / (N / 2)
    V = rfft(data['Potential (AC) (V)'].to_numpy()) / (N / 2)
    I_dict = {}
    V_dict = {}
    k = []
    for i in range(1, max_k+1):
        k_i = np.isclose(freq, i*f, rtol=rtol)

        I_dict['I'+str(i)] = If[k_i]
        V_dict['V'+str(i)] = V[k_i]
        k.append(k_i)

    if freq_domain_visual:
        plot_freq_domain(freq, f, If, V, k)

    return I_dict, V_dict


def fit_Z1(I1, Z1_real, Z1_imag):
    """
    Compute the voltage V1 using the first-harmonic impedance Z1.

    This function calculates the voltage based on the given real and imaginary
    parts of the first-harmonic impedance (Z1), and the current (I).
    The function assumes the impedance is complex (Z1 = Z1_real + j * Z1_imag),
    and the current
    is also complex (I = I_real + j * I_imag).

    Parameters:
    -----------
    I : array-like, shape (2,)
        A two-element array where the first element
        is the real part of the current,
        and the second element is the imaginary part of the current.
    Z1_real : float
        The real part of the first-harmonic impedance (EIS).
    Z1_imag : float
        The imaginary part of the first-harmonic impedance (EIS).

    Returns:
    --------
    ndarray
        A 1D array where the first half contains the real parts of the computed
        voltage (V1) and the second half contains the imaginary parts of V1.
    """
    Z1 = Z1_real + 1j * Z1_imag
    I1c = I1[0] + 1j * I1[1]  # Combine real and imaginary parts of I
    V1 = I1c * Z1  # Voltage V1 calculation using Ohm's law V = I * Z
    # Return concatenated real and imaginary parts
    return np.hstack([V1.real, V1.imag])


def fit_Z2(I1, Z2_real, Z2_imag):
    """
    Compute the voltage V2 using the second-harmonic impedance  Z2.

    This function calculates the voltage based on the given real and imaginary
    parts of the second-harmonic impedance (Z2),
    and the square of the current (I).
    The function assumes the impedance is complex (Z2 = Z2_real + j * Z2_imag),
    and the current is also complex (I = I_real + j * I_imag). The voltage is
    calculated as V2 = Z2 * I^2.

    Parameters:
    -----------
    I : array-like, shape (2,)
        A two-element array where the first element
        is the real part of the current,
        and the second element is the imaginary part of the current.
    Z2_real : float
        The real part of the second-harmonic impedance (2nd-NLEIS).
    Z2_imag : float
        The imaginary part of the second-harmonic impedance (2nd-NLEIS).

    Returns:
    --------
    ndarray
        A 1D array where the first half contains the real parts of the computed
        voltage (V2) and the second half contains the imaginary parts of V2.
    """
    Z2 = Z2_real + 1j * Z2_imag
    I1c = I1[0] + 1j * I1[1]  # Combine real and imaginary parts of I
    V2 = Z2 * I1c ** 2  # Voltage V2 calculation using second-order impedance
    # Return concatenated real and imaginary parts
    return np.hstack([V2.real, V2.imag])


def plot_fit(I1, V1, V2, V1_cal, V2_cal, scaling_factor):
    """Helper function to visualize the data."""
    # The actual visualization need to be determined
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    ax[0].plot(I1.real, V1.real, 'o', color='C0', label="k=1")
    ax[0].plot(I1.real, V2.real * scaling_factor, 'o', color='C1', label="k=2")
    ax[0].plot(I1.real, V1_cal.real, '--', color='C0')
    ax[0].plot(I1.real, V2_cal.real, '--', color='C1')

    ax[1].plot(I1.imag, V1.imag, 'o', color='C0', label="k=1")
    ax[1].plot(I1.imag, V2.imag * scaling_factor, 'o', color='C1', label="k=2")
    ax[1].plot(I1.imag, V1_cal.imag, '--', color='C0')
    ax[1].plot(I1.imag, V2_cal.imag, '--', color='C1')

    ax.legend()
    plt.show()
    return (fig, ax)


def plot_freq_domain(freq, f, If, V, k):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].plot(freq, abs(If))
    ax[1].semilogy(freq, abs(V))
    N = len(k)
    if isinstance(freq, pd.Series) or isinstance(freq, pd.DataFrame):
        ax[0].plot(freq.iloc[k[0]], abs(If.iloc[k[0]]), '.', ms=10)
        for i in range(N):
            ax[1].semilogy(freq.iloc[k[i]], abs(V.iloc[k[i]]), '.', ms=10)

    elif isinstance(freq, np.ndarray):
        ax[0].plot(freq[k[0]], abs(If[k[0]]), '.', ms=10)
        for i in range(N):
            ax[1].semilogy(freq[k[i]], abs(V[k[i]]), '.', ms=10)

    ax[0].set_xlabel('Frequencies, [$\\omega$]')
    ax[0].set_ylabel('Current, [A]')
    ax[1].set_xlabel('Frequencies, [$\\omega$]')
    ax[1].set_ylabel('Voltage, [V]')

    ax[0].text(0.68, 0.9, 'Fundamental Frequency: \n' + str(f)
               + ' Hz', horizontalalignment='center',
               verticalalignment='center', transform=ax[0].transAxes, size=12)
    ax[0].set_xlim([0, (N+1)*f])
    ax[0].set_xticks(np.linspace(0, N*f, N+1))
    ax[0].set_xticklabels(np.arange(0, (N+1), 1))

    ax[1].set_xlim([0, (N+1)*f])
    ax[1].set_xticks(np.linspace(0, N*f, N+1))
    ax[1].set_xticklabels(np.arange(0, (N+1), 1))
    plt.tight_layout()
    plt.show()

    return (fig, ax)


def convert_to_complex(ascStr):
    '''takes Autolab ascii string format for complex current
    or voltage in the frequency domain (after FFT)
    Also turns the nan float values into 0s'''
    if isinstance(ascStr, float):
        return 0
    else:
        ibreak = ascStr.find('I')
        newStr = ascStr[1:ibreak] + ascStr[ibreak+2:-1] + 'j'
        return complex(newStr)
