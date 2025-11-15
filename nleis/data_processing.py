import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import re


def data_truncation(f, Z1, Z2, max_f=10):
    """
    Simple data processing function for EIS and 2nd-NLEIS simultaneously.

    This function processes frequency (`f`), EIS data (`Z1`), and 2nd-NLEIS
    data (`Z2`), removing high-frequency inductance effects and truncating the
    data for 2nd-NLEIS analysis based on the specified maximum frequency.

    Parameters
    ----------
    f : numpy.ndarray
        Array of frequency values.

    Z1 : numpy.ndarray of dtype complex128
        EIS data (complex impedance values).

    Z2 : numpy.ndarray of dtype complex128
        2nd-NLEIS data (complex impedance values).

    max_f : float, optional
        The maximum frequency of interest for 2nd-NLEIS. Frequencies higher
        than `max_f` will be truncated for 2nd-NLEIS data. Default is 10.

    Returns
    -------
    f : numpy.ndarray
        Frequencies after removing high-frequency inductance effects.

    Z1 : numpy.ndarray of dtype complex128
        EIS data with high-frequency inductance removed.

    Z2 : numpy.ndarray of dtype complex128
        2nd-NLEIS data that matches the length of
        high-frequency inductance removed EIS.

    f2_truncated : numpy.ndarray
        Frequencies that are less than
        the specified maximum frequency (`max_f`)
        for 2nd-NLEIS, while removes high frequency inductance

    Z2_truncated : numpy.ndarray of dtype complex128
        2nd-NLEIS data corresponding to the truncated frequency range.

    """

    mask = np.array(Z1.imag) < 0
    f = f[mask]
    Z1 = Z1[mask]
    Z2 = Z2[mask]
    mask1 = np.array(f) < max_f
    f2_truncated = f[mask1]
    Z2_truncated = Z2[mask1]
    return (f, Z1, Z2, f2_truncated, Z2_truncated)


def data_loader(filename, equipment='autolab', fft='instrument', max_k=3,
                multi_current=False, rtol=5e-4, phase_correction=True,
                baseline=True, scaling_factor=1000,
                fit_visualize=False, freq_domain_visual=False):
    """
    Load data and perform FFT processing for EIS and 2nd-NLEIS data.

    Currently supports data from Autolab.

    Parameters
    ----------
    filename : str or list of str
        Path(s) to the CSV file(s) containing the raw data.
        List of str is required if using multi_current mode.
    equipment : str, default 'autolab'
        Specifies the equipment used for data acquisition.
    fft : str, default 'instrument'
        Method used for FFT. Can be 'scipy' or 'instrument'.
    max_k : int, default 3
        Maximum number of harmonics to return in the results.
    multi_current : bool, default False
        If True, process multiple files with different current modulation.
    rtol : float, default 5e-4
        Relative tolerance for finding the frequency components.
    scaling_factor : int, default 1000
        Scaling factor for visualizing 2nd-NLEIS results.
    fit_visualize : bool, default False
        If True, generate plots to visualize the impedance fit
        to multiple current.
        Only available when multi_current = True
    freq_domain_visual : bool, default False
        If True, generate frequency domain current and voltage responses
        at each frequency for the first three harmonics.
        Only available when multi_current = False.

    Returns
    --------
    frequencies : ndarray
        Array of frequency values.
    Z1 : ndarray
        Impedance values for EIS.
    Z2 : ndarray
        Impedance values for 2nd-NLEIS.
    results : DataFrame
        A DataFrame containing the frequencies and
        the current/voltage values in frequency domain
        for up to max_k harmonics.
    """
    if equipment == 'autolab':
        if multi_current:
            if not isinstance(filename, list):
                raise ValueError(
                    "filename should be a list of" +
                    " file paths when multi_current=True")
            # initialize DataFrames for storing the current and voltage values
            df_I1, df_V1, df_V2 = pd.DataFrame(), pd.DataFrame(), \
                pd.DataFrame()
            # process each file in the list
            for index, file in enumerate(filename):
                df = pd.read_csv(file, low_memory=False)
                results = autolab_fft(
                    df, method=fft, rtol=rtol, max_k=max_k,
                    phase_correction=phase_correction, baseline=baseline)
                df_I1[index] = results['I1,[A]']
                df_V1[index] = results['V1,[V]']
                df_V2[index] = results['V2,[V]']

            frequencies = df['Frequency (Hz)'].dropna()
            Z1, Z2 = [], []
            # Obtain impedance value from curve fitting the current and voltage
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
                    Z1[-1], (I1 ** 2) * Z2[-1]

                if fit_visualize:
                    fig, ax = plot_fit(I1, V1, V2, V1_cal,
                                       V2_cal, scaling_factor)
                    plt.show()

            return frequencies.to_numpy(), np.array(Z1), np.array(Z2), results

        else:
            if not isinstance(filename, str):
                raise ValueError(
                    "filename should be a string file " +
                    "paths when multi_current=False")
            df = pd.read_csv(filename, low_memory=False)

            results = autolab_fft(df, method=fft, rtol=rtol, max_k=max_k,
                                  phase_correction=phase_correction,
                                  baseline=baseline,
                                  freq_domain_visual=freq_domain_visual)
            Z1 = results['V1,[V]'] / results['I1,[A]']
            Z2 = results['V2,[V]'] / (results['I1,[A]'] ** 2)

            return (results['freq,[Hz]'].to_numpy(),
                    Z1.to_numpy().astype(np.complex128),
                    Z2.to_numpy().astype(np.complex128), results)


def autolab_fft(df, method='scipy', rtol=5e-4, max_k=3, phase_correction=True,
                baseline=True,
                freq_domain_visual=False):
    """
    Perform FFT on the Autolab time domain or extract Autolab
    frequency domain current and voltage.

    Parameters
    -----------
    df : DataFrame
        Input data from Autolab.
    method : str, default 'scipy'
        FFT method to use. Can be 'scipy' or 'instrument'.
        'scipy' method is used for processing time domain data,
        'instrument' method is used for extracting raw frequency domain data.
    rtol : float, default 5e-4
        Relative tolerance for matching frequency components.
    max_k : int, default 3
        Maximum number of harmonics to return.
    phase_correction : bool, default True
        If True, perform phase correction in the frequency domain
        when method = 'scipy'.
    baseline : bool, default True
        If True, perform baseline subtraction using polynomial fit
        when method = 'scipy'.
    freq_domain_visual: bool, default False
        If True, generate plot for current and voltage in the frequency domain.

    Returns
    --------
    results :
        A DataFrame containing the frequencies and
        the calculated current/voltage values.
    """
    # extract the frequency values
    frequencies = df['Frequency (Hz)'].dropna()
    loc = 0
    # initialize the results DataFrame
    results = pd.DataFrame(columns=[
                           'freq,[Hz]']+[f'I{i+1},[A]' for i in range(max_k)]
                           + [f'V{i+1},[V]' for i in range(max_k)])
    results['freq,[Hz]'] = frequencies

    # extract the time domain data and perform FFT using fft_data function
    # and store the first max_k harmonics current
    # and voltage in the results DataFrame
    if method == 'scipy':

        index = df[df['Time domain (s)'] == 0].index[1]

        for i, f in enumerate(frequencies):
            data = df.iloc[loc:loc+index]
            I, V = fft_data(data, f, max_k=max_k, rtol=rtol,
                            phase_correction=phase_correction,
                            baseline=baseline,
                            freq_domain_visual=freq_domain_visual)
            for j in range(max_k):
                results.loc[i, f'I{j+1},[A]'] = I['I'+str(j+1)]
                results.loc[i, f'V{j+1},[V]'] = V['V'+str(j+1)]

            loc += index
    # extract the frequency domain data from the Autolab output
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
                plt.show()

            loc += index

    return results


def fft_data(data, f, max_k=3, rtol=5e-4, phase_correction=True, baseline=True,
             freq_domain_visual=False):
    """Perform FFT for each frequency
    and return the current and voltage values in frequency domain
    for the first max_k harmonics.

    Parameters
    -----------
    data : DataFrame
        Processed data for each frequency.
    f : float
        Frequency of the measurement.
    max_k : int, default 3
        Maximum number of harmonics to return.
    rtol : float, default 5e-4
        Relative tolerance for matching frequency components.
    phase_correction : bool, default True
        If True, perform phase correction in the frequency domain.
    baseline : bool, default True
        If True, perform baseline subtraction using polynomial fit to
        the surrounding of each harmonic signal using fft_baseline function.
    freq_domain_visual: bool, default False
        If True, generate plot for current and voltage in the frequency domain.

    Returns
    --------
    I_dict: dict
        Dictionary containing the current values for the first max_k harmonics.
    V_dict: dict
        Dictionary containing the voltage values for the first max_k harmonics.
    """
    # perform essential FFT calculation using scipy rfft
    N = len(data)
    sampling_rate = data['Time domain (s)'].iloc[1]
    freq = rfftfreq(N, sampling_rate)

    t_total = data['Time domain (s)'].iloc[-1]-data['Time domain (s)'].iloc[0]
    num_cycle = int(round(t_total*f))

    If = rfft(data['Current (AC) (A)'].to_numpy()) / (N / 2)
    V = rfft(data['Potential (AC) (V)'].to_numpy()) / (N / 2)

    # perform phase correction in frequency domain
    if phase_correction:
        If, V = fft_phase_correction(If, V, num_cycle)

    I_dict = {}
    V_dict = {}
    k = []
    # extract the first max_k harmonics and perform baseline subtraction

    for i in range(1, max_k+1):
        k_i = np.isclose(freq, i*f, rtol=rtol)
        idx = np.where(k_i)[0][0]
        if phase_correction and baseline:
            i_baseline, v_baseline = fft_baseline(freq, If, V, idx)

            I_dict['I'+str(i)] = If[k_i]-i_baseline
            V_dict['V'+str(i)] = V[k_i]-v_baseline

        else:
            I_dict['I'+str(i)] = If[k_i]
            V_dict['V'+str(i)] = V[k_i]
        k.append(k_i)
    # plot the frequency domain current and voltage responses
    if freq_domain_visual:
        plot_freq_domain(freq, f, If, V, k)
        plt.show()

    return I_dict, V_dict


def fit_Z1(I1, Z1_real, Z1_imag):
    """
    Compute the voltage V1 using the first-harmonic impedance Z1.

    This function calculates the voltage based on the given real and imaginary
    parts of the first-harmonic impedance (Z1), and the current (I).
    The function assumes the impedance is complex (Z1 = Z1_real + j * Z1_imag),
    and the current
    is also complex (I = I_real + j * I_imag).

    Parameters
    -----------
    I : numpy.ndarray, dtype=complex128
        The first-harmonic current.
    Z1_real : float
        The real part of the first-harmonic impedance (EIS).
    Z1_imag : float
        The imaginary part of the first-harmonic impedance (EIS).

    Returns
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

    Parameters
    -----------
    I : numpy.ndarray, dtype=complex128
        The first-harmonic current.
    Z2_real : float
        The real part of the second-harmonic impedance (2nd-NLEIS).
    Z2_imag : float
        The imaginary part of the second-harmonic impedance (2nd-NLEIS).

    Returns
    --------
    ndarray
        A 1D array where the first half contains the real parts of the computed
        voltage (V2) and the second half contains the imaginary parts of V2.
    """
    Z2 = Z2_real + 1j * Z2_imag
    I1c = I1[0] + 1j * I1[1]  # Combine real and imaginary parts of I
    # Voltage V2 calculation using second-harmonic impedance
    V2 = Z2 * I1c ** 2
    # Return concatenated real and imaginary parts
    return np.hstack([V2.real, V2.imag])


def plot_fit(I1, V1, V2, V1_cal, V2_cal, scaling_factor):
    """Helper function to visualize the multi current fitting result.
    For valid spectra that has been phase corrected,
    V1 should follows a linear relationship with I1.real,
    while V2 should follows a quadratic relationship with I1.real.

    Parameters
    -----------
    I1 : numpy.ndarray, dtype=complex128
        The first-harmonic current.
    V1 : numpy.ndarray, dtype=complex128
        The first-harmonic voltage.
    V2 : numpy.ndarray, dtype=complex128
        The second-harmonic voltage.
    V1_cal : numpy.ndarray, dtype=complex128
        The calculated first-harmonic voltage.
    V2_cal : numpy.ndarray, dtype=complex128
        The calculated second-harmonic voltage.
    scaling_factor : int
        Scaling factor for visualizing 2nd-NLEIS results.
        Making it comparable to EIS results.

    Returns
    --------
    fig, ax : tuple
        A tuple containing the figure and axis objects.
    """
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    ax[0].plot(I1.real, V1.real, 'o', color='C0', label="k=1")
    ax[0].plot(I1.real, V2.real * scaling_factor, 'o', color='C1', label="k=2")
    ax[0].plot(I1.real, V1_cal.real, '--', color='C0')
    ax[0].plot(I1.real, V2_cal.real * scaling_factor, '--', color='C1')

    ax[1].plot(I1.real, V1.imag, 'o', color='C0', label="k=1")
    ax[1].plot(I1.real, V2.imag * scaling_factor, 'o', color='C1', label="k=2")
    ax[1].plot(I1.real, V1_cal.imag, '--', color='C0')
    ax[1].plot(I1.real, V2_cal.imag * scaling_factor, '--', color='C1')

    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()

    return (fig, ax)


def plot_freq_domain(freq, f, If, V, k):
    '''Plot the frequency domain current and voltage responses
    at each frequency for the first k harmonics.

    Parameters
    -----------
    freq : numpy.ndarray, dtype=float64
        Array of frequency values in freq domain.
    f : float
        Fundamental frequency of the measurement.
    If : numpy.ndarray, dtype=complex128
        Array of current values in frequency domain.
    V : numpy.ndarray, dtype=complex128
        Array of voltage values in frequency domain.
    k : list
        List of indices for the first k harmonics.

    Returns
    --------
    fig, ax : tuple
        A tuple containing the figure and axis objects.
    '''
    # plot the frequency domain current and voltage responses
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].plot(freq, abs(If))
    ax[1].semilogy(freq, abs(V))
    # label the first harmonic current response and
    # the first k harmonics voltage response
    N = len(k)
    if isinstance(freq, pd.Series) or isinstance(freq, pd.DataFrame):
        ax[0].semilogy(freq.iloc[k[0]], abs(If.iloc[k[0]]), '.', ms=10)

        for i in range(N):
            ax[1].semilogy(freq.iloc[k[i]], abs(V.iloc[k[i]]), '.', ms=10)

    elif isinstance(freq, np.ndarray):
        ax[0].semilogy(freq[k[0]], abs(If[k[0]]), '.', ms=10)

        for i in range(N):
            ax[1].semilogy(freq[k[i]], abs(V[k[i]]), '.', ms=10)

    # set the plot labels and limits
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

    return (fig, ax)


def fft_phase_correction(currents, voltages, num_cycle):
    '''
    Phase correction in frequency domain
    based on the first harmonic current response.

    Parameters
    -----------
    currents : numpy.ndarray, dtype=complex128
        Array of current values in frequency domain.
    voltages : numpy.ndarray, dtype=complex128
        Array of voltage values in frequency domain.
    num_cycle : int
        Number of cycles for the measurement.

    Returns
    --------
    I_corrected, V_corrected : tuple
        A tuple containing the phase corrected current and voltage
        values in the frequency domain.

    '''
    # number of data points in frequency domain
    N = len(currents)
    if num_cycle < 0 or num_cycle >= N:
        raise ValueError(
            "num_cycle must be within the range of valid indices (0 to N-1).")

    # reference first harmonic current for phase extraction
    I_phase = currents[num_cycle]

    # initialize arrays for phase corrected currents and voltages
    I_corrected = np.zeros(N, dtype=np.complex128)
    V_corrected = np.zeros(N, dtype=np.complex128)
    # assign zero harmonic values
    I_corrected[0] = currents[0]
    V_corrected[0] = voltages[0]
    # reference phase
    i_phase_reference = 0
    # calculate phase correction
    phase_correction = i_phase_reference - np.angle(I_phase)
    # apply phase correction to all harmonics
    for k in range(1, N):
        I_corrected_phase = np.fmod(
            np.angle(currents[k])+(k/float(num_cycle))*phase_correction,
            2*np.pi)
        I_corrected[k] = np.abs(currents[k])*np.exp(1j*I_corrected_phase)

        V_corrected_phase = np.fmod(
            np.angle(voltages[k])+(k/float(num_cycle))*phase_correction,
            2*np.pi)
        V_corrected[k] = np.abs(voltages[k])*np.exp(1j*V_corrected_phase)

    return (I_corrected, V_corrected)


def fft_baseline(freq, currents, voltages, idx):
    '''Calculate the baseline value for the current and voltage
    at the given frequency index
    using a polynomial fit to the surrounding frequencies
    from idx - 5 to idx - 1 and idx + 2 to idx + 6.

    Parameters
    -----------
    freq : numpy.ndarray, dtype=float64
        Array of frequency values.
    currents : numpy.ndarray, dtype=complex128
        Array of current values.
    voltages : numpy.ndarray, dtype=complex128
        Array of voltage values.
    idx : int
        Index of the frequency of interest.

    Returns
    --------
    I_baseline, V_baseline : tuple
        A tuple containing the baseline current and voltage
        at the given frequency.

    '''
    # polynomial order for the baseline fit
    poly_order = 2
    # frequencies for the baseline fit
    freq_baseline = np.hstack([freq[idx-5:idx-1], freq[idx+2:idx+6]])
    # currents and voltages for the baseline fit
    currents_baseline = np.hstack(
        [currents[idx-5:idx-1], currents[idx+2:idx+6]])
    voltages_baseline = np.hstack(
        [voltages[idx-5:idx-1], voltages[idx+2:idx+6]])
    # polynomial fit for the baseline currents and voltages
    # that consideres both real and imaginary parts
    currents_fit_real = np.polyfit(
        freq_baseline, currents_baseline.real, poly_order)
    currents_fit_imag = np.polyfit(
        freq_baseline, currents_baseline.imag, poly_order)
    voltages_fit_real = np.polyfit(
        freq_baseline, voltages_baseline.real, poly_order)
    voltages_fit_imag = np.polyfit(
        freq_baseline, voltages_baseline.imag, poly_order)

    # initialize polynomial functions the baseline
    p_i_real = np.poly1d(currents_fit_real)
    p_i_imag = np.poly1d(currents_fit_imag)
    p_v_real = np.poly1d(voltages_fit_real)
    p_v_imag = np.poly1d(voltages_fit_imag)

    # find the index of the frequency component
    k_idx = freq[idx]

    # calculate the baseline currents and voltages at the given frequency
    I_baseline = p_i_real(k_idx)+1j*p_i_imag(k_idx)
    V_baseline = p_v_real(k_idx)+1j*p_v_imag(k_idx)

    return (I_baseline, V_baseline)


def convert_to_complex(ascStr):
    '''takes Autolab ascii string format for complex current
    or voltage in the frequency domain (after FFT)
    Also turns the nan float values into 0s'''
    if isinstance(ascStr, float):
        # If the input is a float (often something like 'NaN'), return 0
        return 0
    else:
        # Otherwise, parse the ASCII string of the form '(1 + 2I)'

        ibreak = ascStr.find('I')
        newStr = ascStr[1:ibreak] + ascStr[ibreak+2:-1] + 'j'
        return complex(newStr)


def thd(df):
    """
    Calculate the Total Harmonic Distortion (THD) for current and voltage
    in the frequency domain.

    To calculate the THD, please set max_k to 10 in the dataloader() function.

    Parameters
    -----------
    df : DataFrame
        DataFrame returned by the data_loader function.

    Returns
    --------
    thd_I, thd_V : tuple
        A tuple containing the THD values for current and voltage.

    Notes
    -----

    .. math::

        THD = \\sqrt{\\frac{\\sum_{n=2}^{10} Y_n^2}{Y_1}},

    where :math:`Y_1` is the fundamental frequency component
    and :math:`Y_n` (for :math:`n \\geq 2`) are the higher harmonic components.

    """
    current_columns = [col for col in df.columns if re.match(r'I', col)]
    voltage_columns = [col for col in df.columns if re.match(r'V', col)]

    thd_I = np.sum(abs(df[current_columns[1:]])**2,
                   axis=1)**0.5/abs(df[current_columns[0]])
    thd_V = np.sum(abs(df[voltage_columns[1:]])**2,
                   axis=1)**0.5/abs(df[voltage_columns[0]])

    return thd_I, thd_V
