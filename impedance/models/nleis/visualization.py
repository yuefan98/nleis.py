import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

### These are all slightly modified versin of the visualization code. Can be seamlessly inegrated into impedance.py in the future.

def plot_altair(data_dict,k = 1, units = 'Ω', size=400, background='#FFFFFF'):
    """ Plots impedance as an interactive Nyquist/Bode plot using altair

        Parameters
        ----------
        data_dict: dict
            dictionary with keys
            'f': frequencies
            'Z': impedance
            'fmt': {'-' for a line, else circles}
        size: int
            size in pixels of Nyquist height/width
        background: str
            hex color string for chart background (default = '#FFFFFF')

        Returns
        -------
        chart: altair.Chart
    """

    Z_df = pd.DataFrame(columns=['f', 'z_real', 'z_imag', 'kind', 'fmt'])
    for kind in data_dict.keys():
        f = data_dict[kind]['f']
        Z = data_dict[kind]['Z']
        fmt = data_dict[kind].get('fmt', 'o')

        df = pd.DataFrame({'f': f, 'z_real': Z.real, 'z_imag': Z.imag,
                           'kind': kind, 'fmt': fmt})

        Z_df = pd.concat([Z_df,df])
        # Z_df.append(df)

    range_x = max(Z_df['z_real']) - min(Z_df['z_real'])
    range_y = max(-Z_df['z_imag']) - min(-Z_df['z_imag'])

    rng = max(range_x, range_y)

    min_x = min(Z_df['z_real'])
    max_x = min(Z_df['z_real']) + rng
    min_y = min(-Z_df['z_imag'])
    max_y = min(-Z_df['z_imag']) + rng

    nearest = alt.selection_single(on='mouseover', nearest=True,
                                   empty='none', fields=['f'])
    ## potential future improvement
    # nearest = alt.selection_point(on='mouseover', nearest=True,
    #                                empty='none', fields=['f'])

    fmts = Z_df['fmt'].unique()
    nyquists, bode_mags, bode_phss = [], [], []
    if '-' in fmts:
        df = Z_df.groupby('fmt').get_group('-')
        ##### These are changed to introduce harmonics and units
        nyquist = alt.Chart(df).mark_line().encode(
            x=alt.X('z_real:Q', axis=alt.Axis(title="Z{}' [{}]".format(k, units)),
                    scale=alt.Scale(domain=[min_x, max_x],
                                    nice=False, padding=5),sort=None),
            y=alt.Y('neg_z_imag:Q', axis=alt.Axis(title="-Z{}'' [{}]".format(k, units)),
                    scale=alt.Scale(domain=[min_y, max_y],
                                    nice=False, padding=5),sort=None),
            color='kind:N'
        ).properties(
            height=size,
            width=size
        ).transform_calculate(
            neg_z_imag='-datum.z_imag'
        )

        bode = alt.Chart(df).mark_line().encode(
            alt.X('f:Q', axis=alt.Axis(title="f [Hz]"),
                  scale=alt.Scale(type='log', nice=False),sort=None),
            color='kind:N'
        ).properties(
            width=size,
            height=size/2 - 25
        ).transform_calculate(
            mag="sqrt(pow(datum.z_real,2) + pow(datum.z_imag,2))",
            neg_phase="-(180/PI)*atan2(datum.z_imag,datum.z_real)"
        )

        bode_mag = bode.encode(y=alt.Y('mag:Q',
                                       axis=alt.Axis(title="|Z{}|' [{}]".format(k, units)),sort=None))
        bode_phs = bode.encode(y=alt.Y('neg_phase:Q',
                                       axis=alt.Axis(title="-ϕ [°]"),sort=None))

        nyquists.append(nyquist)
        bode_mags.append(bode_mag)
        bode_phss.append(bode_phs)
        nyquists

    if 'o' in fmts:
        df = Z_df.groupby('fmt').get_group('o')

        nyquist = alt.Chart(df).mark_circle().encode(
            x=alt.X('z_real:Q', axis=alt.Axis(title="Z{}' [{}]".format(k, units)),
                    scale=alt.Scale(domain=[min_x, max_x],
                                    nice=False, padding=5),sort=None),
            y=alt.Y('neg_z_imag:Q', axis=alt.Axis(title="-Z{}'' [{}]".format(k, units)),
                    scale=alt.Scale(domain=[min_y, max_y],
                                    nice=False, padding=5),sort=None),
            size=alt.condition(nearest, alt.value(80), alt.value(30)),
            color=alt.Color('kind:N', legend=alt.Legend(title='Legend'))
        ).add_selection(
            nearest
        ).properties(
            height=size,
            width=size
        ).transform_calculate(
            neg_z_imag='-datum.z_imag'
        ).interactive()

        bode = alt.Chart(df).mark_circle().encode(
            alt.X('f:Q', axis=alt.Axis(title="f [Hz]"),
                  scale=alt.Scale(type='log', nice=False),sort=None),
            size=alt.condition(nearest, alt.value(80), alt.value(30)),
            color='kind:N'
        ).add_selection(
            nearest
        ).properties(
            width=size,
            height=size/2 - 25
        ).transform_calculate(
            mag="sqrt(pow(datum.z_real,2) + pow(datum.z_imag,2))",
            neg_phase="-(180/PI)*atan2(datum.z_imag,datum.z_real)"
        ).interactive()

        bode_mag = bode.encode(y=alt.Y('mag:Q',
                                       axis=alt.Axis(title="|Z{}|' [{}]".format(k, units)),sort=None))
        bode_phs = bode.encode(y=alt.Y('neg_phase:Q',
                                       axis=alt.Axis(title="-ϕ [°]"),sort=None))

        nyquists.append(nyquist)
        bode_mags.append(bode_mag)
        bode_phss.append(bode_phs)

    full_bode = alt.layer(*bode_mags) & alt.layer(*bode_phss)

    return (full_bode | alt.layer(*nyquists)).configure(background=background)
def plot_first(ax, Z, scale=1, fmt='.-', **kwargs):
    """ Plots impedance as a Nyquist plot using matplotlib

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            axes on which to plot the nyquist plot
        Z: np.array of complex numbers
            impedance data
        scale: float
            the scale for the axes
        units: string
            units for :math:`Z(\\omega)`
        fmt: string
            format string passed to matplotlib (e.g. '.-' or 'o')

        Other Parameters
        ----------------
        **kwargs : `matplotlib.pyplot.Line2D` properties, optional
            Used to specify line properties like linewidth, line color,
            marker color, and line labels.

        Returns
        -------
        ax: matplotlib.axes.Axes
    """

    ax.plot(np.real(Z), -np.imag(Z), fmt, **kwargs)

    # Make the axes square
    ax.set_aspect(aspect = 1 ,anchor='C',adjustable='datalim')

    # Set the labels to -imaginary vs real
    ax.set_xlabel(r'$\tilde{Z}_{1}^{\prime}(\omega)$' +
                          ' [$\Omega$]', fontsize=20)
    ax.set_ylabel(r'$-\tilde{Z}_{1}^{\prime\prime}(\omega)$' +
                          ' [$\Omega$]', fontsize=20)

    # Make the tick labels larger
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Change the number of labels on each axis to five
    ax.locator_params(axis='x', nbins=5, tight=True)
    ax.locator_params(axis='y', nbins=5, tight=True)

    # Add a light grid
    ax.grid(visible=True, which='major', axis='both', alpha=.5)

    # Change axis units to 10**log10(scale) and resize the offset text
    #ax.ticklabel_format(style='sci', axis='both')
    
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)
    limits = -np.log10(scale)
    if limits != 0:
        ax.ticklabel_format(style='sci', axis='both',
                            scilimits=(limits, limits))
    y_offset = ax.yaxis.get_offset_text()
    y_offset.set_size(18)
    t = ax.xaxis.get_offset_text()
    t.set_size(18)


    return ax

def plot_second(ax, Z, scale=1, fmt='.-', **kwargs):
    """ Plots impedance as a Nyquist plot using matplotlib

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            axes on which to plot the nyquist plot
        Z: np.array of complex numbers
            impedance data
        scale: float
            the scale for the axes
        units: string
            units for :math:`Z(\\omega)`
        fmt: string
            format string passed to matplotlib (e.g. '.-' or 'o')

        Other Parameters
        ----------------
        **kwargs : `matplotlib.pyplot.Line2D` properties, optional
            Used to specify line properties like linewidth, line color,
            marker color, and line labels.

        Returns
        -------
        ax: matplotlib.axes.Axes
    """

    ax.plot(np.real(Z), -np.imag(Z), fmt, **kwargs)

    # Make the axes square
    ax.set_aspect(aspect = 1 ,anchor='C',adjustable='datalim')

    # Set the labels to -imaginary vs real
    ax.set_xlabel(r'$\tilde{Z}_{2}^{\prime}(\omega)$' +
                          ' [$\Omega / A$]', fontsize=20)
    ax.set_ylabel(r'$-\tilde{Z}_{2}^{\prime\prime}(\omega)$' +
                          ' [$\Omega / A$]', fontsize=20)
    # Make the tick labels larger
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Change the number of labels on each axis to five
    ax.locator_params(axis='x', nbins=5, tight=True)
    ax.locator_params(axis='y', nbins=5, tight=True)

    # Add a light grid
    ax.grid(visible=True, which='major', axis='both', alpha=.5)

    # Change axis units to 10**log10(scale) and resize the offset text
    
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1))
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter) 


    limits = -np.log10(scale)
    if limits != 0:
        ax.ticklabel_format(style='scientific', axis='both',
                            scilimits=(limits, limits))
    y_offset = ax.yaxis.get_offset_text()
    y_offset.set_size(18)
    t = ax.xaxis.get_offset_text()
    t.set_size(18)

    return ax

