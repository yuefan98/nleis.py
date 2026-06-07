import numpy as np
from scipy.special import iv
from scipy.linalg import solve_banded
from scipy import constants
from impedance.models.circuits.elements import circuit_elements, \
    ElementError, OverwriteError

F = constants.physical_constants['Faraday constant'][0]
R = constants.R
T = 298.15

# element function adopted from impedance.py for better documentation


def element(num_params, units, overwrite=False):
    """

    decorator to store metadata for a circuit element

    Parameters
    ----------
    num_params : int
        number of parameters for an element
    units : list of str
        list of units for the element parameters
    overwrite : bool (default False)
        if true, overwrites any existing element; if false,
        raises OverwriteError if element name already exists.

    """

    def decorator(func):
        def wrapper(p, f):
            typeChecker(p, f, func.__name__, num_params)
            return func(p, f)

        wrapper.num_params = num_params
        wrapper.units = units
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__

        global circuit_elements
        if func.__name__ in ["s", "p", "d"]:
            raise ElementError("cannot redefine elements 's' (series)" +
                               "or 'p' (parallel)" + " or 'd' (difference)")
        elif func.__name__ in circuit_elements and not overwrite:
            raise OverwriteError(
                f"element {func.__name__} already exists. " +
                "If you want to overwrite the existing element," +
                "use `overwrite=True`."
            )
        else:
            circuit_elements[func.__name__] = wrapper

        # Adding numpy to circuit_elements for proper evaluation with
        # numpy>=2.0.0 because the scalar representation was changed.
        # "Scalars are now printed as np.float64(3.0) rather than just 3.0."
        # https://numpy.org/doc/2.0/release/2.0.0-notes.html
        # #representation-of-numpy-scalars-changed
        circuit_elements["np"] = np

        return wrapper

    return decorator


def d(difference):
    '''
    This function calculates the impedance difference between two electrodes
    In a two electrode cell, subtract the positive electrode 2nd-NLEIS from
    the negative electrode 2nd-NLEIS to get the cell response.

    Notes
    -----

    .. math::

        Z_2 = Z_2^{+} - Z_2^{-}

    '''

    z = len(difference[0])*[0 + 0*1j]
    z += difference[0]
    z += -difference[-1]
    return z


# manually add difference (d) operators to circuit elements w/o metadata
circuit_elements['d'] = d


@element(num_params=2, units=['Ohm', 'F'])
def RC(p, f):
    """

    EIS: Randles circuit (charge transfer only)

    Notes
    -----

    .. math::

        \\tilde{Z_1} = \\frac{R_{ct}}{1 + \\omega^{*}  j}

    and

    .. math::

        \\omega^{*} = \\omega R_{ct} C_{dl}


    **Parameters:**

    .. math::

        p[0] = R_{ct}; \\;
        p[1] = C_{dl}; \\;

    """
    ω = 2*np.pi * np.array(f)
    Rct, Cdl = p[0], p[1]

    ω_star = ω*Rct*Cdl

    return Rct / (1 + ω_star*1j)


@element(num_params=3, units=['Ohm', 'F', '-'])
def RCn(p, f):
    '''

    2nd-NLEIS: Nonlinear Randles circuit
    (charge transfer only) from Ji et al. [1]

    Notes
    -----

    .. math::
        \\tilde{Z_2} = \\frac{-\\varepsilon f R_{ct}^2}
        {1 + 4\\omega^{*}  j - 5{\\omega^{*}}^2 - 2{\\omega^{*}}^3 j}

    and

    .. math::

        \\omega^{*} = \\omega R_{ct} C_{dl}


    **Parameters:**

    .. math::

        p[0] = R_{ct}; \\;
        p[1] = C_{dl}; \\;
        p[2] = ε; \\;

    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    '''
    ω = 2*np.pi * np.array(f)
    Rct, Cdl, ε = p[0], p[1], p[2]

    ω_star = ω*Rct*Cdl

    return -0.5 * ε*F/(R*T)*Rct**2 / (1 + 4*ω_star*1j -
                                      5*ω_star**2 - 2*ω_star**3*1j)


@element(num_params=4, units=['Ohms', 'F', 'Ohms', 's'])
def RCD(p, f):
    '''

    EIS: Randles circuit with diffusion
    in a bounded thin film electrode from Ji et al. [1]

    Notes
    -----

    .. math::
        \\tilde{Z_1} = \\frac{R_{ct}}{\\frac{R_{ct}}
        {R_{ct} + \\tilde{Z}_{D,1}} + j\\omega^{*}}

    and

    .. math::

        \\omega^{*} = \\omega R_{ct} C_{dl}

    and

    .. math::

        Z_{D,1} = \\frac{A_w \\coth\\left(\\sqrt{j\\omega\\tau}
        \\right)}{\\sqrt{j\\omega\\tau}}

    **Parameters:**

    .. math::

        p[0] = R_{ct}; \\;
        p[1] = C_{dl}; \\;
        p[2] = A_{w}; \\;
        p[3] = τ; \\;

    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    '''
    ω = 2*np.pi * np.array(f)
    Rct, Cdl, Aw, τd = p[0], p[1], p[2], p[3]

    sqrt_1j_ω_τd = np.sqrt(1j*ω*τd)
    tanh_1j_ω_τd = np.tanh(sqrt_1j_ω_τd)
    Zd = Aw / (sqrt_1j_ω_τd * tanh_1j_ω_τd)

    ω_star = ω*Rct*Cdl
    Z = Rct / (Rct / (Rct + Zd) + 1j*ω_star)
    return (Z)


@element(num_params=6, units=['Ohms', 'F', 'Ohms', 's', '1/V', '-'])
def RCDn(p, f):
    '''

    2nd-NLEIS: Nonlinear Randles circuit with diffusion
    in a bounded thin film electrode from Ji et al. [1]

    Notes
    -----

    .. math::

        \\tilde{Z_2} = \\frac{R_{ct}}{\\left(j2\\omega^{*}
        + \\frac{R_{ct}}{\\tilde{Z}_{D,2} + R_{ct}}\\right)}
        \\frac{\\left[ \\kappa
        \\left( \\frac{\\tilde{Z}_{D,1}}{\\tilde{Z}_{D,1}
        + R_{ct}} \\right)^2 - \\varepsilon f
        \\left( \\frac{R_{ct}}{\\tilde{Z}_{D,1}
        + R_{ct}} \\right)^2 \\right]}{\\tilde{Z}_{D,2} + R_{ct}}
        \\left( \\frac{R_{ct}}{\\frac{R_{ct}}{R_{ct}
        + \\tilde{Z}_{D,1}} + j\\omega^{*}} \\right)^2

    and

    .. math::

        \\omega^{*} = \\omega R_{ct} C_{dl}

    and

    .. math::

        Z_{D,1} = \\frac{A_w \\coth\\left(\\sqrt{j\\omega\\tau}
        \\right)}{\\sqrt{j\\omega\\tau}}

    and

    .. math::

        Z_{D,2} = \\frac{A_w \\coth\\left(\\sqrt{j2\\omega\\tau}
        \\right)}{\\sqrt{j2\\omega\\tau}}


    **Parameters:**

    .. math::
        p[0] = R_{ct}; \\;
        p[1] = C_{dl}; \\;
        p[2] = A_{w}; \\;
        p[3] = τ; \\;
        p[4] = κ; \\;
        p[5] = ε; \\;

    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    '''

    ω = 2 * np.pi * np.array(f)
    Rct, Cdl, Aw, τd, κ, ε = p[0], p[1], p[2], p[3], p[4], p[5]

    sqrt_1j_ω_τd = np.sqrt(1j * ω * τd)
    tanh_1j_ω_τd = np.tanh(sqrt_1j_ω_τd)
    Zd1 = Aw / (sqrt_1j_ω_τd * tanh_1j_ω_τd)

    sqrt_1j_2ω_τd = np.sqrt(1j * 2 * ω * τd)
    tanh_1j_2ω_τd = np.tanh(sqrt_1j_2ω_τd)
    Zd2 = Aw / (sqrt_1j_2ω_τd * tanh_1j_2ω_τd)

    ω_star = ω * Rct * Cdl
    y1 = Rct / (Zd1 + Rct)
    y2 = Zd1 / (Zd1 + Rct)

    Z1 = Rct / (y1 + 1j * ω_star)
    const = ((Rct * κ * y2**2) - Rct * ε * F / (R * T) * y1**2) / (Zd2 + Rct)

    Z2 = (const * Z1**2) / (2 * ω_star * 1j + Rct / (Zd2 + Rct))

    return 0.5 * Z2


@element(num_params=4, units=['Ohms', 'F', 'Ohms', 's'])
def RCS(p, f):
    '''

    EIS: Randles circuit with diffusion
    diffusion into a spherical electrode from Ji et al. [1]

    Notes
    -----

    .. math::
        \\tilde{Z_1} = \\frac{R_{ct}}{\\frac{R_{ct}}
        {R_{ct} + \\tilde{Z}_{D,1}} + j\\omega^{*}}

    and

    .. math::

        \\omega^{*} = \\omega R_{ct} C_{dl}

    and

    .. math::

        Z_{D,1} = \\frac{A_{w} \\tanh\\left( \\sqrt{j\\omega\\tau}
          \\right)}{\\sqrt{j\\omega\\tau}
            - \\tanh\\left( \\sqrt{j\\omega\\tau} \\right)}


    **Parameters:**

    .. math::

        p[0] = R{ct}; \\;
        p[1] = C_{dl}; \\;
        p[2] = A_{w}; \\;
        p[3] = τ; \\;

    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representation
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    '''
    ω = 2 * np.pi * np.array(f)
    Rct, Cdl, Aw, τd = p[0], p[1], p[2], p[3]

    sqrt_1j_ω_τd = np.sqrt(1j * ω * τd)
    tanh_1j_ω_τd = np.tanh(sqrt_1j_ω_τd)
    Zd = Aw*tanh_1j_ω_τd / (sqrt_1j_ω_τd - tanh_1j_ω_τd)

    ω_star = ω * Rct * Cdl
    Z = Rct / (Rct / (Rct + Zd) + 1j * ω_star)
    return Z


@element(num_params=6, units=['Ohms', 'F', 'Ohms', 's', '1/V', '-'])
def RCSn(p, f):
    '''

    2nd-NLEIS: Nonlinear Randles circuit with diffusion
    diffusion into a spherical electrode from Ji et al. [1]

    Notes
    -----

    .. math::

        \\tilde{Z_2} = \\frac{R_{ct}}{\\left(j2\\omega^{*}
        + \\frac{R_{ct}}{\\tilde{Z}_{D,2} + R_{ct}}\\right)}
        \\frac{\\left[ \\kappa
        \\left( \\frac{\\tilde{Z}_{D,1}}{\\tilde{Z}_{D,1}
        + R_{ct}} \\right)^2 - \\varepsilon f
        \\left( \\frac{R_{ct}}{\\tilde{Z}_{D,1}
        + R_{ct}} \\right)^2 \\right]}{\\tilde{Z}_{D,2} + R_{ct}}
        \\left( \\frac{R_{ct}}{\\frac{R_{ct}}{R_{ct}
        + \\tilde{Z}_{D,1}} + j\\omega^{*}} \\right)^2

    and

    .. math::

        \\omega^{*} = \\omega R_{ct} C_{dl}

    and

    .. math::

        Z_{D,1} = Z_{D,1} = \\frac{A_{w} \\tanh\\left( \\sqrt{j\\omega\\tau}
          \\right)}{\\sqrt{j\\omega\\tau}
            - \\tanh\\left( \\sqrt{j\\omega\\tau} \\right)}

    and

    .. math::

        Z_{D,2} = \\frac{A_{w} \\tanh\\left( \\sqrt{j2\\omega\\tau}
          \\right)}{\\sqrt{j2\\omega\\tau}
            - \\tanh\\left( \\sqrt{j2\\omega\\tau} \\right)}


    **Parameters:**

    .. math::

        p[0] = R_{ct}; \\;
        p[1] = C_{dl}; \\;
        p[2] = A_{w}; \\;
        p[3] = τ; \\;
        p[4] = κ; \\;
        p[5] = ε; \\;

    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    '''

    ω = 2 * np.pi * np.array(f)
    Rct, Cdl, Aw, τd, κ, ε = p[0], p[1], p[2], p[3], p[4], p[5]

    sqrt_1j_ω_τd = np.sqrt(1j * ω * τd)
    tanh_1j_ω_τd = np.tanh(sqrt_1j_ω_τd)
    Zd1 = Aw * tanh_1j_ω_τd / (sqrt_1j_ω_τd - tanh_1j_ω_τd)

    sqrt_1j_2ω_τd = np.sqrt(1j * 2 * ω * τd)
    tanh_1j_2ω_τd = np.tanh(sqrt_1j_2ω_τd)
    Zd2 = Aw * tanh_1j_2ω_τd / (sqrt_1j_2ω_τd - tanh_1j_2ω_τd)

    ω_star = ω * Rct * Cdl
    y1 = Rct / (Zd1 + Rct)
    y2 = Zd1 / (Zd1 + Rct)

    Z1 = Rct / (y1 + 1j * ω_star)
    const = ((Rct * κ * y2**2) - Rct * ε * F / (R * T) * y1**2) / (Zd2 + Rct)

    Z2 = (const * Z1**2) / (2 * ω_star * 1j + Rct / (Zd2 + Rct))

    return 0.5 * Z2


@element(num_params=3, units=['Ohms', 'Ohms', 'F'])
def TP(p, f):
    '''

    EIS: Porous electrode with high conductivity matrix (charge transfer only)
    from Ji et al. [1]

    Notes
    -----

    .. math::

        Z_1 = \\frac{R_{\\text{pore}} \\coth(\\beta_1)}{\\beta_1}

    where

    .. math::

        \\beta_1 = \\left( j \\omega C_{\\text{dl}} R_{\\text{pore}} +
        \\frac{R_{\\text{pore}}}{R_{\\text{ct}}} \\right)^{\\frac{1}{2}}

    **Parameters:**

    .. math::

        p[0] = R_{pore}; \\;
        p[1] = R_{ct}; \\;
        p[2] = C_{dl}; \\;

    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    '''

    ω = 2 * np.pi * np.array(f)
    Rpore, Rct, Cdl = p[0], p[1], p[2]

    beta = np.sqrt(1j * ω * Rpore * Cdl + Rpore / Rct)

    Z = Rpore / (beta * np.tanh(beta))
    return Z


@element(num_params=4, units=['Ohms', 'Ohms', 'F', '-'])
def TPn(p, f):
    """

    2nd-NLEIS: Porous electrode with high conductivity matrix
    (charge transfer only) from Ji et al. [1]

    Notes
    -----

    .. math::

        Z_2 = \\frac{ε f R_{\\text{pore}}^3}{R_{\\text{ct}}
        (\\beta_1 \\sinh(\\beta_1))^2}
        \\left[ \\left( \\frac{\\beta_1 \\sinh(2\\beta_1)}
        {\\beta_2(\\beta_2 - 2\\beta_1)
        (\\beta_2 + 2\\beta_1)} \\coth(\\beta_2) \\right) - \n
        \\left( \\frac{\\cosh(2\\beta_1)}{2(\\beta_2 - 2\\beta_1)
        (\\beta_2 + 2\\beta_1)} + \\frac{1}{2\\beta_2^2} \\right) \\right]

    where

    .. math::

        \\beta_1 = \\left( j \\omega C_{\\text{dl}} R_{\\text{pore}}
        + \\frac{R_{\\text{pore}}}{R_{\\text{ct}}} \\right)^{\\frac{1}{2}}

    and

    .. math::

        \\beta_2 = \\left( j 2\\omega C_{\\text{dl}} R_{\\text{pore}}
        + \\frac{R_{\\text{pore}}}{R_{\\text{ct}}} \\right)^{\\frac{1}{2}}

    **Parameters:**

    .. math::
        p[0] = R_{pore}; \\;
        p[1] = R_{ct}; \\;
        p[2] = C_{dl}; \\;
        p[3] = ε; \\;

    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    """

    ω = 2 * np.pi * np.array(f)
    Rpore, Rct, Cdl, ε = p[0], p[1], p[2], p[3]
    b1 = np.sqrt(1j * ω * Rpore * Cdl + Rpore / Rct)
    b2 = np.sqrt(1j * 2 * ω * Rpore * Cdl + Rpore / Rct)

    sinh1 = np.where(b1 < 100, np.sinh(b1), 1e10)
    sinh2 = np.where(b1 < 100, np.sinh(2 * b1), 1e10)
    cosh2 = np.where(b1 < 100, np.cosh(2 * b1), 1e10)

    mf = ((Rpore ** 3) / Rct) * ε * (F / (R * T)) / ((b1 * sinh1) ** 2)
    part1 = (b1 / b2) * sinh2 / ((b2 ** 2 - 4 * b1 ** 2) * np.tanh(b2))
    part2 = -cosh2 / (2 * (b2 ** 2 - 4 * b1 ** 2)) - 1 / (2 * b2 ** 2)
    Z = mf * (part1 + part2)

    return 0.5 * Z


@element(num_params=5, units=['Ohms', 'Ohms', 'F', 'Ohms', 's'])
def TDP(p, f):
    """

    EIS: Porous electrode with high conductivity matrix
    and planar diffusion into platelet-like particles from Ji et al. [1]

    Notes
    -----

    .. math::

        Z_1 = \\frac{R_{\\text{pore}} \\coth(\\beta_1^D)}{\\beta_1^D}

    where

    .. math::

        \\beta_1^D = \\left( j\\omega C_{\\text{dl}} R_{\\text{pore}}
        + \\frac{R_{\\text{pore}}}
        {Z_{D,1} + R_{\\text{ct}}} \\right)^{\\frac{1}{2}}

    and

    .. math::

        Z_{D,1} = \\frac{A_{w} \\coth\\left( \\sqrt{j\\omega\\tau}
        \\right)}{\\sqrt{j\\omega\\tau}}


    **Parameters:**

    .. math::

        p[0] = R_{pore}; \\;
        p[1] = R_{ct}; \\;
        p[2] = C_{dl}; \\;
        p[3] = A_{w}; \\;
        p[4] = τ; \\;


    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    """
    ω = 2 * np.pi * np.array(f)
    Rpore, Rct, Cdl, Aw, τd = p[0], p[1], p[2], p[3], p[4]

    sqrt_1j_ω_τd = np.sqrt(1j * ω * τd)
    tanh_1j_ω_τd = np.tanh(sqrt_1j_ω_τd)
    Zd = Aw / (sqrt_1j_ω_τd * tanh_1j_ω_τd)

    beta = np.sqrt(1j * ω * Rpore * Cdl + Rpore / (Zd + Rct))
    tanh_beta = np.tanh(beta)

    Z = Rpore / (beta * tanh_beta)
    return Z


@element(num_params=7, units=['Ohms', 'Ohms', 'F', 'Ohms', 's', '1/V', '-'])
def TDPn(p, f):
    """

    2nd-NLEIS: A macrohomogeneous porous electrode model with planar diffusion
    and zero solid resistivity from Ji et al. [1]

    Notes
    -----

    .. math::

        Z_2 = \\frac{ε f R_{\\text{pore}}^3}{R_{\\text{ct}}
        (\\beta_1^D \\sinh(\\beta_1^D))^2}
        \\left[ \\left( \\frac{\\beta_1^D \\sinh(2\\beta_1^D)}
        {\\beta_2^D(\\beta_2^D - 2\\beta_1^D)
        (\\beta_2^D + 2\\beta_1^D)} \\coth(\\beta_2^D) \\right) - \n
        \\left( \\frac{\\cosh(2\\beta_1^D)}{2(\\beta_2^D - 2\\beta_1^D)
        (\\beta_2^D + 2\\beta_1^D)} +
        \\frac{1}{2{\\beta_2^D}^2} \\right) \\right]

    where

    .. math::

        \\beta_1^D = \\left( j2\\omega C_{\\text{dl}} R_{\\text{pore}} +
        \\frac{R_{\\text{pore}}}{\\tilde{Z}_{D,1}
        + R_{\\text{ct}}} \\right)^{\\frac{1}{2}}

    and

    .. math::

        \\beta_2^D = \\left( j2\\omega C_{\\text{dl}} R_{\\text{pore}} +
        \\frac{R_{\text{pore}}}{\\tilde{Z}_{D,2}
        + R_{\\text{ct}}} \\right)^{\\frac{1}{2}}

    and

    .. math::

        Z_{D,1} = \\frac{A_{w} \\coth\\left( \\sqrt{j\\omega\\tau}
        \\right)}{\\sqrt{j\\omega\\tau}}

    and

    .. math::

        Z_{D,2} = \\frac{A_{w} \\coth\\left( \\sqrt{j2\\omega\\tau}
        \\right)}{\\sqrt{j2\\omega\\tau}}


    **Parameters:**

    .. math::
        p[0] = R_{pore}; \\;
        p[1] = R{ct}; \\;
        p[2] = C_{dl}; \\;
        p[3] = A_{w}; \\;
        p[4] = τ; \\;
        p[5] = κ; \\;
        p[6] = ε; \\;

    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    """

    ω = 2 * np.pi * np.array(f)

    Rpore, Rct, Cdl, Aw, τd, κ, ε = p[0], p[1], p[2], p[3], p[4], p[5], p[6]

    sqrt_1j_ω_τd = np.sqrt(1j * ω * τd)
    sqrt_1j_2ω_τd = np.sqrt(1j * 2 * ω * τd)

    Zd1 = Aw / (sqrt_1j_ω_τd * np.tanh(sqrt_1j_ω_τd))
    Zd2 = Aw / (sqrt_1j_2ω_τd * np.tanh(sqrt_1j_2ω_τd))

    y1 = Rct / (Zd1 + Rct)
    y2 = Zd1 / (Zd1 + Rct)

    b1 = np.sqrt(1j * ω * Rpore * Cdl + Rpore / (Zd1 + Rct))
    b2 = np.sqrt(1j * 2 * ω * Rpore * Cdl + Rpore / (Zd2 + Rct))

    sinh1 = np.where(np.abs(b1) < 100, np.sinh(b1), 1e10)
    sinh2 = np.where(np.abs(b1) < 100, np.sinh(2 * b1), 1e10)
    cosh2 = np.where(np.abs(b1) < 100, np.cosh(2 * b1), 1e10)

    const = -((Rct * κ * y2**2) - Rct * ε * (F / (R * T)) * y1**2)/(Zd2+Rct)
    mf = ((Rpore**3) * const / Rct) / ((b1 * sinh1)**2)
    part1 = (b1 / b2) * sinh2 / ((b2**2 - 4 * b1**2) * np.tanh(b2))
    part2 = -cosh2 / (2 * (b2**2 - 4 * b1**2)) - 1 / (2 * b2**2)
    Z = mf * (part1 + part2)

    return 0.5 * Z


@element(num_params=5, units=['Ohms', 'Ohms', 'F', 'Ohms', 's'])
def TDS(p, f):
    """

    EIS: porous electrode with high conductivity matrix and
    diffusion into spherical particles from Ji et al. [1]

    Notes
    -----

    .. math::

        Z_1 = \\frac{R_{\\text{pore}} \\coth(\\beta_1^D)}{\\beta_1^D}

    where

    .. math::

        \\beta_1^D = \\left( j\\omega C_{\\text{dl}} R_{\\text{pore}}
        + \\frac{R_{\\text{pore}}}
        {Z_{D,1} + R_{\\text{ct}}} \\right)^{\\frac{1}{2}}

    and

    .. math::

        Z_{D,1} = \\frac{A_{w} \\tanh\\left( \\sqrt{j\\omega\\tau}
        \\right)}{\\sqrt{j\\omega\\tau}
        - \\tanh\\left( \\sqrt{j\\omega\\tau} \\right)}


    **Parameters:**

    .. math::

        p[0] = Rpore; \\;
        p[1] = Rct; \\;
        p[2] = Cdl; \\;
        p[3] = Aw; \\;
        p[4] = τd; \\;

    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.


    """
    ω = 2 * np.pi * np.array(f)
    Rpore, Rct, Cdl, Aw, τd = p[0], p[1], p[2], p[3], p[4]

    sqrt_1j_ω_τd = np.sqrt(1j * ω * τd)
    tanh_1j_ω_τd = np.tanh(sqrt_1j_ω_τd)
    Zd = Aw * tanh_1j_ω_τd / (sqrt_1j_ω_τd - tanh_1j_ω_τd)

    beta = np.sqrt(1j * ω * Rpore * Cdl + Rpore / (Zd + Rct))

    Z = Rpore / (beta * np.tanh(beta))
    return Z


@element(num_params=7, units=['Ohms', 'Ohms', 'F', 'Ohms', 's', '1/V', '-'])
def TDSn(p, f):
    """

    2nd-NLEIS: porous electrode with high conductivity matrix and
    diffusion into spherical particles from Ji et al. [1]

    Notes
    -----

    .. math::

        Z_2 = \\frac{ε f R_{\\text{pore}}^3}{R_{\\text{ct}}
        (\\beta_1^D \\sinh(\\beta_1^D))^2}
        \\left[ \\left( \\frac{\\beta_1^D \\sinh(2\\beta_1^D)}
        {\\beta_2^D(\\beta_2^D - 2\\beta_1^D)
        (\\beta_2^D + 2\\beta_1^D)} \\coth(\\beta_2^D) \\right) - \n
        \\left( \\frac{\\cosh(2\\beta_1^D)}{2(\\beta_2^D - 2\\beta_1^D)
        (\\beta_2^D + 2\\beta_1^D)} +
        \\frac{1}{2{\\beta_2^D}^2} \\right) \\right]

    where

    .. math::

        \\beta_1^D = \\left( j2\\omega C_{\\text{dl}} R_{\\text{pore}} +
        \\frac{R_{\\text{pore}}}{\\tilde{Z}_{D,1}
        + R_{\\text{ct}}} \\right)^{\\frac{1}{2}}

    and

    .. math::

        \\beta_2^D = \\left( j2\\omega C_{\\text{dl}} R_{\\text{pore}} +
        \\frac{R_{\text{pore}}}{\\tilde{Z}_{D,2}
        + R_{\\text{ct}}} \\right)^{\\frac{1}{2}}

    and

    .. math::

        Z_{D,1} = \\frac{A_{w} \\tanh\\left( \\sqrt{j\\omega\\tau}
        \\right)}{\\sqrt{j\\omega\\tau}
        - \\tanh\\left( \\sqrt{j\\omega\\tau} \\right)}

    and

    .. math::

        Z_{D,2} = \\frac{A_{w} \\tanh\\left( \\sqrt{j2\\omega\\tau}
        \\right)}{\\sqrt{j2\\omega\\tau}
        - \\tanh\\left( \\sqrt{j2\\omega\\tau} \\right)}

    **Parameters:**

    .. math::

        p[0] = Rpore; \\;
        p[1] = Rct; \\;
        p[2] = Cdl; \\;
        p[3] = Aw; \\;
        p[4] = τ; \\;
        p[5] = κ; \\;
        p[6] = ε; \\;

    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.


    """

    ω = 2 * np.pi * np.array(f)
    Rpore, Rct, Cdl, Aw, τd, κ, ε = p[0], p[1], p[2], p[3], p[4], p[5], p[6]

    sqrt_1j_ω_τd = np.sqrt(1j * ω * τd)
    sqrt_1j_2ω_τd = np.sqrt(1j * 2 * ω * τd)
    tanh_1j_ω_τd = np.tanh(sqrt_1j_ω_τd)
    tanh_1j_2ω_τd = np.tanh(sqrt_1j_2ω_τd)

    Zd1 = Aw * tanh_1j_ω_τd / (sqrt_1j_ω_τd - tanh_1j_ω_τd)
    Zd2 = Aw * tanh_1j_2ω_τd / (sqrt_1j_2ω_τd - tanh_1j_2ω_τd)

    y1 = Rct / (Zd1 + Rct)
    y2 = Zd1 / (Zd1 + Rct)

    b1 = np.sqrt(1j * ω * Rpore * Cdl + Rpore / (Zd1 + Rct))
    b2 = np.sqrt(1j * 2 * ω * Rpore * Cdl + Rpore / (Zd2 + Rct))

    sinh1 = np.where(b1 < 100, np.sinh(b1), 1e10)
    sinh2 = np.where(b1 < 100, np.sinh(2 * b1), 1e10)
    cosh2 = np.where(b1 < 100, np.cosh(2 * b1), 1e10)

    const = -((Rct * κ * y2**2) - Rct * ε * (F / (R * T)) * y1**2)/(Zd2+Rct)
    mf = ((Rpore**3) * const / Rct) / ((b1 * sinh1)**2)
    part1 = (b1 / b2) * sinh2 / ((b2**2 - 4 * b1**2) * np.tanh(b2))
    part2 = -cosh2 / (2 * (b2**2 - 4 * b1**2)) - 1 / (2 * b2**2)
    Z = mf * (part1 + part2)

    return 0.5 * Z


@element(num_params=5, units=['Ohms', 'Ohms', 'F', 'Ohms', 's'])
def TDC(p, f):
    """

    EIS: porous electrode with high conductivity matrix and
    diffusion into cylindrical particles from Ji et al. [1]

    Notes
    -----

    .. math::

        Z_1 = \\frac{R_{\\text{pore}} \\coth(\\beta_1^D)}{\\beta_1^D}

    where

    .. math::

        \\beta_1^D = \\left( j\\omega C_{\\text{dl}} R_{\\text{pore}}
        + \\frac{R_{\\text{pore}}}
        {Z_{D,1} + R_{\\text{ct}}} \\right)^{\\frac{1}{2}}

    and

    .. math::

        Z_{D,1} = A_w \\frac{I_0\\left(\\sqrt{j \\omega \\tau}\\right)}
        {\\sqrt{j \\omega \\tau} I_1\\left(\\sqrt{j \\omega \\tau}\\right)}

    **Parameters:**

    .. math::
        p[0] = R_{pore}; \\;
        p[1] = R_{ct}; \\;
        p[2] = C_{dl}; \\;
        p[3] = A_{w}; \\;
        p[4] = τ; \\;

    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    """
    ω = 2 * np.pi * np.array(f)
    Rpore, Rct, Cdl, Aw, τd = p[0], p[1], p[2], p[3], p[4]

    sqrt_1j_ω_τd = np.sqrt(1j * ω * τd)

    i01 = iv(0, sqrt_1j_ω_τd)
    i11 = iv(1, sqrt_1j_ω_τd)

    i01 = np.where(sqrt_1j_ω_τd < 100, i01, 1e20)
    i11 = np.where(sqrt_1j_ω_τd < 100, i11, 1e20)

    Zd = Aw * i01 / (sqrt_1j_ω_τd * i11)

    beta = np.sqrt(1j * ω * Rpore * Cdl + Rpore / (Zd + Rct))
    Z = Rpore / (beta * np.tanh(beta))
    return Z


@element(num_params=7, units=['Ohms', 'Ohms', 'F', 'Ohms', 's', '1/V', '-'])
def TDCn(p, f):
    """

    2nd-NLEIS: porous electrode with high conductivity matrix and
    diffusion into cylindrical particles from Ji et al. [1]

    Notes
    -----

    .. math::

        Z_2 = \\frac{ε f R_{\\text{pore}}^3}{R_{\\text{ct}}
        (\\beta_1^D \\sinh(\\beta_1^D))^2}
        \\left[ \\left( \\frac{\\beta_1^D \\sinh(2\\beta_1^D)}
        {\\beta_2^D(\\beta_2^D - 2\\beta_1^D)
        (\\beta_2^D + 2\\beta_1^D)} \\coth(\\beta_2^D) \\right) - \n
        \\left( \\frac{\\cosh(2\\beta_1^D)}{2(\\beta_2^D - 2\\beta_1^D)
        (\\beta_2^D + 2\\beta_1^D)} +
        \\frac{1}{2{\\beta_2^D}^2} \\right) \\right]

    where

    .. math::

        \\beta_1^D = \\left( j2\\omega C_{\\text{dl}} R_{\\text{pore}} +
        \\frac{R_{\\text{pore}}}{\\tilde{Z}_{D,1}
        + R_{\\text{ct}}} \\right)^{\\frac{1}{2}}

    and

    .. math::

        \\beta_2^D = \\left( j2\\omega C_{\\text{dl}} R_{\\text{pore}} +
        \\frac{R_{\text{pore}}}{\\tilde{Z}_{D,2}
        + R_{\\text{ct}}} \\right)^{\\frac{1}{2}}

    and

    .. math::

        Z_{D,1} = A_w \\frac{I_0\\left(\\sqrt{j \\omega \\tau}\\right)}
        {\\sqrt{j \\omega \\tau} I_1\\left(\\sqrt{j \\omega \\tau}\\right)}

    and

    .. math::

        Z_{D,2} = A_w \\frac{I_0\\left(\\sqrt{j 2\\omega \\tau}\\right)}
        {\\sqrt{j 2\\omega \\tau} I_1\\left(\\sqrt{j 2\\omega \\tau}\\right)}

    **Parameters:**

    .. math::

        p[0] = R_{pore}; \\;
        p[1] = R_{ct}; \\;
        p[2] = C_{dl}; \\;
        p[3] = A_{w}; \\;
        p[4] = τ; \\;
        p[5] = κ; \\;
        p[6] = ε; \\;

    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    """

    ω = 2 * np.pi * np.array(f)
    Rpore, Rct, Cdl, Aw, τd, κ, ε = p[0], p[1], p[2], p[3], p[4], p[5], p[6]

    sqrt_1j_ω_τd = np.sqrt(1j * ω * τd)
    sqrt_1j_2ω_τd = np.sqrt(1j * 2 * ω * τd)

    i01 = np.where(sqrt_1j_ω_τd < 100, iv(0, sqrt_1j_ω_τd), 1e20)
    i11 = np.where(sqrt_1j_ω_τd < 100, iv(1, sqrt_1j_ω_τd), 1e20)
    i02 = np.where(sqrt_1j_2ω_τd < 100, iv(0, sqrt_1j_2ω_τd), 1e20)
    i12 = np.where(sqrt_1j_2ω_τd < 100, iv(1, sqrt_1j_2ω_τd), 1e20)

    Zd1 = Aw * i01 / (sqrt_1j_ω_τd * i11)
    Zd2 = Aw * i02 / (sqrt_1j_2ω_τd * i12)

    y1 = Rct / (Zd1 + Rct)
    y2 = Zd1 / (Zd1 + Rct)

    b1 = np.sqrt(1j * ω * Rpore * Cdl + Rpore / (Zd1 + Rct))
    b2 = np.sqrt(1j * 2 * ω * Rpore * Cdl + Rpore / (Zd2 + Rct))

    sinh1 = np.where(b1 < 100, np.sinh(b1), 1e10)
    sinh2 = np.where(b1 < 100, np.sinh(2 * b1), 1e10)
    cosh2 = np.where(b1 < 100, np.cosh(2 * b1), 1e10)

    const = -((Rct * κ * y2**2) - Rct * ε * (F / (R * T)) * y1**2) / (Zd2+Rct)
    mf = ((Rpore**3) * const / Rct) / ((b1 * sinh1)**2)
    part1 = (b1 / b2) * sinh2 / ((b2**2 - 4 * b1**2) * np.tanh(b2))
    part2 = -cosh2 / (2 * (b2**2 - 4 * b1**2)) - 1 / (2 * b2**2)
    Z = mf * (part1 + part2)

    return 0.5 * Z

##################################################################
# TLM Model
##################################################################


def A_matrices_TLMn(N, Rpore, Z12t):
    """
    Construct the matrix `Ax` for the TLMn model

    Parameters
    ----------
    N : int
        Number of circuit elements
    Rpore : float
        Pore electrolyte resistance
    Z12t : np.complex128
        The single element impedance at 2ω

    Returns
    -------
    Ax : np.ndarray
        The matrix `Ax` for the TLMn model

    """

    Ax = np.zeros((N, N), dtype=np.complex128)
    # Construct matrix `A`
    for i in range(N - 1):
        for j in range(N - 1 - i):
            # construct the Rpore term
            Ax[i, j] = (N - 1 - i - j) * Rpore
        # construct the Rpore Z12t term
        Ax[i, 0] += Z12t
        Ax[i, N - 1 - i] -= Z12t
    # construct the last row of the matrix
    Ax[-1, :] = 1
    return (Ax)


def solve_banded_tridiagonal(diagonal, off_diagonal, rhs):
    """
    Solve a tridiagonal linear system using SciPy's banded format.

    Parameters
    ----------
    diagonal : np.ndarray
        Main diagonal of the tridiagonal matrix.
    off_diagonal : complex
        Constant upper and lower off-diagonal value.
    rhs : np.ndarray
        Right-hand side vector.

    Returns
    -------
    x : np.ndarray
        Solution of the tridiagonal system.

    """
    n = len(rhs)
    ab = np.zeros((3, n), dtype=np.complex128)
    ab[0, 1:] = off_diagonal
    ab[1, :] = diagonal
    ab[2, :-1] = off_diagonal
    return solve_banded((1, 1), ab, rhs)


def solve_tlm_current_distribution(Zb, Rpore, Ztot, N):
    """
    Solve the first-harmonic current distribution for a TLM.

    The dense matrix used in Eq. 14 of Ji and Schwartz [1] has the
    structure

    .. math::

        A^{(N)}_{kl} = Z_{1,b}^{(N)} \\delta_{kl}
        + R_{\\text{pore},N} \\min(k, l),
        \\quad k,l=1,\\ldots,N

    or

    .. math::

        A^{(N)} = Z_{1,b}^{(N)} I + R_{\\text{pore},N} B,
        \\quad B_{kl}=\\min(k,l).

    The inverse of :math:`B` is tridiagonal,

    .. math::

        B^{-1} =
        \\begin{bmatrix}
        2 & -1 & 0 & \\cdots & 0 \\\\
        -1 & 2 & -1 & \\cdots & 0 \\\\
        0 & -1 & 2 & \\cdots & 0 \\\\
        \\vdots & \\vdots & \\vdots & \\ddots & -1 \\\\
        0 & 0 & 0 & -1 & 1
        \\end{bmatrix}.

    Multiplying Eq. 14 by :math:`B^{-1}` gives the equivalent
    tridiagonal system

    .. math::

        \\left(Z_{1,b}^{(N)} B^{-1} + R_{\\text{pore},N} I\\right)
        \\theta_{1}^{(N)}
        = B^{-1} Z_{1}^{(N)}.

    :math:`Z_{1}^{(N)}` is the constant vector
    whose entries are the total impedance calculated with the corresponding
    TLM functions.

    Parameters
    ----------
    Zb : np.ndarray
        Branch impedance at the fundamental frequency.
    Rpore : float
        Discretized pore electrolyte resistance.
    Ztot : np.ndarray
        Total impedance plus final pore electrolyte resistance.

    N : int
        Number of TLM elements.

    Returns
    -------
    I1 : np.ndarray
        The first-harmonic current distribution with shape
        ``(len(Zb), N)``.

    References
    ----------
    [1] Y. Ji and D. T. Schwartz, Extending Transmission Line Models
    to Second-Harmonic Nonlinear Electrochemical Impedance Spectra,
    J. Electrochem. Soc. 173, 053501 (2026).

    """
    len_freq = len(Zb)
    I1 = np.zeros((len_freq, N), dtype=np.complex128)

    q_diagonal = np.full(N, 2, dtype=np.complex128)
    q_diagonal[-1] = 1
    rhs = np.zeros(N, dtype=np.complex128)

    for freq_idx in range(len_freq):
        Zb_freq = Zb[freq_idx]
        diagonal = Zb_freq * q_diagonal + Rpore
        rhs[0] = Ztot[freq_idx]
        rhs[1:] = 0
        I1[freq_idx, :] = solve_banded_tridiagonal(
            diagonal, -Zb_freq, rhs
        )

    return I1


def solve_tlm_second_harmonic_current(N, Rpore, Z12t, rhs):
    """
    Solve the second-harmonic TLM current distribution from Eq. 18.
    An alternative to the dense matrix solve is presented here to
    speed up the solution for large N.

    Following the notation of Ji and Schwartz [1], let

    .. math::

        x_i = \\left(\\theta_2^{(N)}\\right)_i,
        \\quad i=1,\\ldots,N,

    .. math::

        Z = Z_{1@2ω,b}^{(N)},
        \\quad R = R_{\\text{pore},N}.

    The code accepts a general right-hand side whose final entry is
    :math:`s`; for the physical zero second-harmonic flux boundary
    condition in Eq. 15, :math:`s=0`:

    .. math::

        \\sum_{i=1}^{N} x_i = s.

    For :math:`m=1,\\ldots,N-1`, the Eq. 18 voltage-balance rows can
    be written as

    .. math::

        Z(x_1 - x_{m+1})
        + R
        \\sum_{j=1}^{m}(m+1-j)x_j = E_m,

    followed by the boundary row

    .. math::

        \\sum_{k=1}^{N} x_k = s.

    The dense matrix in Eq. 18 stores these
    :math:`E_m` rows in reverse order:

    .. math::

        E_m = \\text{rhs}_{N-1-m},
        \\quad m=1,\\ldots,N-1.

    Taking differences of the first :math:`N-1` equations removes the
    cumulative pore-resistance kernel. The first transformed row is
    simply the :math:`m=1` voltage balance:

    .. math::

        (Z + R)x_1 - Zx_2 = E_1,

    For the remaining rows, use the second finite difference
    :math:`F_m - 2F_{m-1} + F_{m-2}`, where

    .. math::

        F_m =
        Z(x_1 - x_{m+1})
        + R \\sum_{j=1}^{m}(m+1-j)x_j.

    The paper defines :math:`E_m` only for :math:`m=1,\\ldots,N-1`.
    For the :math:`m=2` transformed row, we introduce the auxiliary
    :math:`m=0` extension

    .. math::

        F_0 = Z(x_1 - x_1)
        + R \\sum_{j=1}^{0}(\\cdots) = 0,

    so the compatible auxiliary right-hand-side value is
    :math:`E_0=0`. This is not an additional physical equation from
    the paper; it is only a notational device that makes the finite
    difference formula uniform.

    Therefore the second transformed row is

    .. math::

        -Zx_1 + (2Z + R)x_2 - Zx_3 = E_2 - 2E_1,

    and the general rows are

    .. math::

        -Zx_{m-1} + (2Z + R)x_m - Zx_{m+1}
        = E_m - 2E_{m-1} + E_{m-2},
        \\quad m=3,\\ldots,N-1.

    Equivalently, the same formula can be used for
    :math:`m=2,\\ldots,N-1` if :math:`E_0=0` is understood as the
    auxiliary value above. This gives a bordered tridiagonal system

    together with

    .. math::

        \\sum_{k=1}^{N} x_k = s.

    Letting :math:`\\mathbf{u}=(x_1,\\ldots,x_{N-1})` and
    :math:`v=x_N`, this is

    .. math::

        M\\mathbf{u} + \\mathbf{q}v = \\mathbf{d},
        \\quad \\mathbf{1}^T\\mathbf{u} + v = s.

    The tridiagonal matrix :math:`M` is

    .. math::

        M =
        \\begin{bmatrix}
        Z+R & -Z & 0 & \\cdots & 0 \\\\
        -Z & 2Z+R & -Z & \\cdots & 0 \\\\
        0 & -Z & 2Z+R & \\cdots & 0 \\\\
        \\vdots & \\vdots & \\vdots & \\ddots & -Z \\\\
        0 & 0 & 0 & -Z & 2Z+R
        \\end{bmatrix},

    and the border vector is

    .. math::

        \\mathbf{q} =
        \\begin{bmatrix}
        0 & 0 & \\cdots & -Z
        \\end{bmatrix}^T.

    The transformed right-hand side is

    .. math::

        d_1 = E_1,

    .. math::

        d_m = E_m - 2E_{m-1} + E_{m-2},
        \\quad m=2,\\ldots,N-1.

    The border is eliminated with two tridiagonal solves,

    .. math::

        \\mathbf{y}=M^{-1}\\mathbf{d},
        \\quad \\mathbf{w}=M^{-1}\\mathbf{q},

    .. math::

        v = \\frac{s - \\mathbf{1}^T\\mathbf{y}}
        {1 - \\mathbf{1}^T\\mathbf{w}},
        \\quad \\mathbf{u}=\\mathbf{y}-\\mathbf{w}v.

    The full solution returned by this helper is then

    .. math::

        \\mathbf{x} =
        \\begin{bmatrix}
        \\mathbf{u} \\\\
        v
        \\end{bmatrix},

    in the same order as the dense Eq. 18 solve.

    Parameters
    ----------
    N : int
        Number of TLM elements.
    Rpore : float
        Discretized pore electrolyte resistance.
    Z12t : complex
        Branch impedance at twice the fundamental frequency.
    rhs : np.ndarray
        Right-hand side vector in the same row order expected by
        :func:`A_matrices_TLMn`.

    Returns
    -------
    I2 : np.ndarray
        Solution in the same order as ``np.linalg.solve(Ax, rhs)`` for
        ``Ax = A_matrices_TLMn(N, Rpore, Z12t)``.

    References
    ----------
    [1] Y. Ji and D. T. Schwartz, Extending Transmission Line Models
    to Second-Harmonic Nonlinear Electrochemical Impedance Spectra,
    J. Electrochem. Soc. 173, 053501 (2026).

    """
    if N == 1:
        return np.array([rhs[-1]], dtype=np.complex128)

    E = np.zeros(N, dtype=np.complex128)
    E[1:] = rhs[N-2::-1]

    d = np.empty(N - 1, dtype=np.complex128)
    d[0] = E[1]
    if N > 2:
        d[1:] = E[2:] - 2 * E[1:-1] + E[:-2]

    diagonal = np.full(N - 1, 2 * Z12t + Rpore, dtype=np.complex128)
    diagonal[0] = Z12t + Rpore
    off_diagonal = -Z12t

    q = np.zeros(N - 1, dtype=np.complex128)
    q[-1] = off_diagonal

    y = solve_banded_tridiagonal(diagonal, off_diagonal, d)
    w = solve_banded_tridiagonal(diagonal, off_diagonal, q)

    v = (rhs[-1] - np.sum(y)) / (1 - np.sum(w))
    u = y - w * v
    return np.append(u, v)


@element(num_params=6, units=['Ohm', 'Ohm', 'F', 'Ohm', 'F', '-'])
def TLM(p, f):
    """

    EIS： General discrete transmission line model built  Randles circuit

    Notes
    -----


    **Parameters:**

    .. math::

        p[0] = R_{pore}; \\;
        p[1] = R_{ct, bulk}; \\;
        p[2] = C_{dl, bulk}; \\;
        p[3] = R_{ct, surface}; \\;

    .. math::

        p[4] = C_{dl, surface}; \\;
        p[5] = N (\\text{number of circuit element}); \\;


    """
    frequencies = np.array(f)
    N = int(p[5])

    Rct = p[1] * N
    Cdl = p[2] / N
    Rpore = p[0] / N
    Rs = p[3] * N
    Cs = p[4] / N

    Z1b = RC([Rct, Cdl], frequencies)
    Z1s = RC([Rs, Cs], frequencies)
    Zran = Z1b + Z1s

    Req = np.copy(Zran)
    inv_Zran = 1 / Zran
    for _ in range(1, N):
        Req = 1 / ((1 / (Req + Rpore)) + inv_Zran)

    return Req


@element(num_params=8, units=['Ohm', 'Ohm', 'F', '-', 'Ohm', 'F', '-', '-'])
def TLMn(p, f):
    """

    2nd-NLEIS: Second harmonic nonlinear discrete transmission line model
    built based on the nonlinear Randles circuit from Ji et al. [1]

    Notes
    -----

    .. math::


    **Parameters:**

    .. math::

        p[0] = R_{pore}; \\;
        p[1] = R_{ct,bulk}; \\;
        p[2] = C_{dl,bulk}; \\;
        p[3] = R_{ct,surface}; \\;

    .. math::

        p[4] = C_{dl,surface}; \\;
        p[5] = N (\\text{number of circuit element}); \\;
        p[6] = ε_{bulk}; \\;
        p[7] = ε_{surface}; \\;



    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    """
    # calculate the current fraction (1st harmonic)
    I1 = mTi(p[0:6], f)

    frequencies = np.array(f)

    N = int(p[5])

    Rpore = p[0] / N
    Rct = p[1] * N
    Cdl = p[2] / N
    Rs = p[3] * N
    Cs = p[4] / N

    εb = p[6]
    εs = p[7]

    Z1b = RC([Rct, Cdl], frequencies)
    Z1s = RC([Rs, Cs], frequencies)
    Z2b = RCn([Rct, Cdl, εb], frequencies)
    Z2s = RCn([Rs, Cs, εs], frequencies)
    Z1b2t = RC([Rct, Cdl], 2 * frequencies)
    Z1s2t = RC([Rs, Cs], 2 * frequencies)

    Z1 = Z1b + Z1s
    Z2 = Z2b + Z2s
    Z12t = Z1b2t + Z1s2t

    if N == 1:
        return Z2

    if N == 2:
        sum1 = Z1**2 / (2 * Z1 + Rpore)**2
        sum2 = (Z12t * Rpore + Rpore**2) / \
            ((2 * Z12t + Rpore) * (2 * Z1 + Rpore))
        Z = (sum1 + sum2) * Z2
        return Z
    len_freq = len(frequencies)
    Z = np.zeros(len_freq, dtype=np.complex128)
    for freq_idx in range(len_freq):
        Ii = I1[freq_idx]
        rhs = np.zeros(N, dtype=np.complex128)
        rhs[:-1] = (Ii[:-1]**2 - Ii[-1]**2) * Z2[freq_idx]
        I2 = solve_tlm_second_harmonic_current(
            N, Rpore, Z12t[freq_idx], rhs
        )
        Z[freq_idx] = Z2[freq_idx] * Ii[0]**2 + I2[-1] * Z12t[freq_idx]

    return Z


@element(num_params=6, units=['Ohm', 'Ohm', 'F', 'Ohm', 'F', '-'])
def mTi(p, f):
    """

    EIS: current distribution of  discrete transmission line model
    built based on the  Randles circuit from Ji et al. [1]

    Notes
    -----

    .. math::



    **Parameters:**

    .. math::

        p[0] = R_{pore}; \\;
        p[1] = R_{ct,bulk}; \\;
        p[2] = C_{dl,bulk}; \\;
        p[3] = R_{ct,surface}; \\;

    .. math::

        p[4] = C_{dl,surface}; \\;
        p[5] = N (\\text{number of circuit element}); \\;


    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    """

    N = int(p[5])
    frequencies = np.array(f)

    Rct = p[1]*N
    Cdl = p[2]/N
    Rpore = p[0]/N
    Rs = p[3]*N
    Cs = p[4]/N
    Z1b = RC([Rct, Cdl], frequencies)
    Z1s = RC([Rs, Cs], frequencies)
    Zran = Z1b + Z1s
    Req = np.copy(Zran)
    inv_Zran = 1 / Zran
    for i in range(1, N):

        Req = 1 / ((1 / (Req + Rpore)) + inv_Zran)

    Req = Req+Rpore

    return solve_tlm_current_distribution(Zran, Rpore, Req, N)


@element(num_params=8, units=['Ohm', 'Ohm', 'F', 'Ohm', 's', 'Ohm', 'F', '-'])
def TLMS(p, f):
    """

    EIS: General discrete transmission line model
    built based on the Randles circuit
    with spherical diffusion from Ji et al.[1]

    Notes
    -----

    .. math::



    **Parameters:**

    .. math::
        p[0] = R_{pore}; \\;
        p[1] = R_{ct,bulk}; \\;
        p[2] = C_{dl,bulk}; \\;
        p[3] = A_{w,bulk}; \\;
        p[4] = τ_{bulk}; \\;

    .. math::

        p[5] = R_{ct,surface}; \\;
        p[6] = C_{dl,surface}; \\;
        p[7] = N (\\text{number of circuit element}); \\;


    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    """

    N = int(p[7])
    frequencies = np.array(f)

    Rpore = p[0]/N
    Rct = p[1]*N
    Cdl = p[2]/N
    Aw = p[3]*N
    τd = p[4]
    Rs = p[5]*N
    Cs = p[6]/N
    Z1b = RCS([Rct, Cdl, Aw, τd], frequencies)

    Z1s = RC([Rs, Cs], frequencies)

    Zran = Z1b + Z1s
    Req = np.copy(Zran)
    inv_Zran = 1 / Zran
    for _ in range(1, N):

        Req = 1 / ((1 / (Req + Rpore)) + inv_Zran)

    return (Req)


@element(num_params=11, units=['Ohm', 'Ohm', 'F', 'Ohm', 's', 'Ohm', 'F', '-',
                               '1/V', '-', '-'])
def TLMSn(p, f):
    """

    2nd-NLEIS: Second harmonic nonlinear discrete transmission line model
    built based on the Randles circuit
    with spherical diffusion from Ji et al. [1]

    Notes
    -----

    .. math::



    **Parameters:**

    .. math::
        p[0] = R_{pore}; \\;
        p[1] = R_{ct,bulk}; \\;
        p[2] = C_{dl,bulk}; \\;
        p[3] = A_{w,bulk}; \\;
        p[4] = τ_{bulk}; \\;
        p[5] = R_{ct,surface}; \\;


    .. math::

        p[6] = C_{dl,surface}; \\;
        p[7] = N (\\text{number of circuit element}); \\;
        p[8] = κ_{bulk}; \\;

    .. math::
        p[9] = ε_{bulk}; \\;
        p[10] = ε_{surface}; \\;



    """
    frequencies = np.array(f)

    # calculate the current fraction (1st harmonic)
    I1 = mTiS(p[0:8], frequencies)

    N = int(p[7])

    Rpore = p[0]/N
    Rct = p[1]*N
    Cdl = p[2]/N
    Aw = p[3]*N
    τd = p[4]
    Rs = p[5]*N
    Cs = p[6]/N
    κ = p[8]
    εb = p[9]
    εs = p[10]

    Z1b = RCS([Rct, Cdl, Aw, τd], frequencies)
    Z1s = RC([Rs, Cs], frequencies)
    Z2b = RCSn([Rct, Cdl, Aw, τd, κ, εb], frequencies)
    Z2s = RCn([Rs, Cs, εs], frequencies)
    Z1b2t = RCS([Rct, Cdl, Aw, τd], 2*frequencies)
    Z1s2t = RC([Rs, Cs], 2*frequencies)

    Z1 = Z1b + Z1s
    Z2 = Z2b + Z2s
    Z12t = Z1b2t + Z1s2t

    if N == 1:
        return (Z2)

    if N == 2:
        sum1 = Z1**2 / (2*Z1+Rpore)**2
        sum2 = (Z12t*Rpore+Rpore**2) / ((2*Z12t+Rpore)*(2*Z1+Rpore))
        Z = (sum1+sum2)*Z2
        return (Z)

    len_freq = len(frequencies)
    Z = np.zeros(len_freq, dtype=np.complex128)

    for freq_idx in range(len_freq):
        Ii = I1[freq_idx]
        Z12t_freq = Z12t[freq_idx]
        Z2_freq = Z2[freq_idx]
        rhs = np.zeros(N, dtype=np.complex128)
        rhs[:-1] = (Ii[:-1]**2 - Ii[-1]**2) * Z2_freq
        I2 = solve_tlm_second_harmonic_current(
            N, Rpore, Z12t_freq, rhs
        )
        Z[freq_idx] = Z2_freq*Ii[0]**2 + I2[-1]*Z12t_freq

    return (Z)


@element(num_params=8, units=['Ohm', 'Ohm', 'F', 'Ohm', 's', 'Ohm', 'F', '-'])
def mTiS(p, f):
    """

    EIS: current distribution of nonlinear discrete transmission line model
    built based on the Randles circuit
    with spherical diffusion from Ji et al. [1]

    Notes
    -----

    .. math::

    **Parameters:**

    .. math::
        p[0] = R_{pore}; \\;
        p[1] = R_{ct,bulk}; \\;
        p[2] = C_{dl,bulk}; \\;
        p[3] = A_{w,bulk}; \\;
        p[4] = τ_{bulk}; \\;
        p[5] = R_{ct,surface}; \\;


    .. math::

        p[6] = C_{dl,surface}; \\;
        p[7] = N (\\text{number of circuit element}); \\;


    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    """
    N = int(p[7])
    frequencies = f

    Rpore = p[0]/N
    Rct = p[1]*N
    Cdl = p[2]/N
    Aw = p[3]*N
    τd = p[4]
    Rs = p[5]*N
    Cs = p[6]/N

    Z1b = RCS([Rct, Cdl, Aw, τd], frequencies)

    Z1s = RC([Rs, Cs], frequencies)
    Zran = Z1b + Z1s
    Req = np.copy(Zran)
    inv_Zran = 1 / Zran
    for _ in range(1, N):

        Req = 1 / ((1 / (Req + Rpore)) + inv_Zran)

    Req = Req+Rpore
    return solve_tlm_current_distribution(Zran, Rpore, Req, N)


@element(num_params=11, units=['Ohm', 'Ohm', 'F', 'Ohm', 's', 'Ohm', 'F', '-',
                               '1/V', '-', '-'])
def mTiSn(p, f):
    """

    2nd-NLEIS: nonlinear current distribution of
    nonlinear discrete transmission line model
    built based on the Randles circuit
    with spherical diffusion from Ji et al. [1]

    Notes
    -----

    .. math::



    **Parameters:**

    .. math::
        p[0] = R_{pore}; \\;
        p[1] = R_{ct,bulk}; \\;
        p[2] = C_{dl,bulk}; \\;
        p[3] = A_{w,bulk}; \\;
        p[4] = τ_{bulk}; \\;
        p[5] = R_{ct,surface}; \\;


    .. math::

        p[6] = C_{dl,surface}; \\;
        p[7] = N (\\text{number of circuit element}); \\;
        p[8] = κ_{bulk}; \\;

    .. math::
        p[9] = ε_{bulk}; \\;
        p[10] = ε_{surface}; \\;



    """
    frequencies = np.array(f)
    # calculate the current fraction (1st harmonic)
    I1 = mTiS(p[0:8], frequencies)

    N = int(p[7])

    Rpore = p[0]/N
    Rct = p[1]*N
    Cdl = p[2]/N
    Aw = p[3]*N
    τd = p[4]
    Rs = p[5]*N
    Cs = p[6]/N
    κ = p[8]
    eb = p[9]
    es = p[10]

    Z2b = RCSn([Rct, Cdl, Aw, τd, κ, eb], frequencies)
    Z2s = RCn([Rs, Cs, es], frequencies)
    Z1b2t = RCS([Rct, Cdl, Aw, τd], 2*frequencies)
    Z1s2t = RC([Rs, Cs], 2*frequencies)

    Z2 = Z2b + Z2s
    Z12t = Z1b2t + Z1s2t
    len_freq = len(frequencies)
    if N == 1:
        return (0)

    I2 = np.zeros((len_freq, N), dtype=np.complex128)

    if N == 2:

        I2[:, 0] = Z2*Rpore / (2*Z12t+Rpore)**2
        I2[:, 1] = -Z2*Rpore / (2*Z12t+Rpore)**2

        return (I2)

    for freq_idx in range(0, len_freq):
        Ii = I1[freq_idx]
        rhs = np.zeros(N, dtype=np.complex128)
        rhs[:-1] = (Ii[:-1]**2 - Ii[-1]**2) * Z2[freq_idx]
        # Reverse the order to display the correct result from small to
        # larger N, matching the historical dense implementation.
        I2[freq_idx, :] = solve_tlm_second_harmonic_current(
            N, Rpore, Z12t[freq_idx], rhs
        )[::-1]

    return (I2)


@element(num_params=8, units=['Ohm', 'Ohm', 'F', 'Ohm', 's', 'Ohm', 'F', '-'])
def TLMD(p, f):
    """

    EIS: general discrete transmission line model
    built based on the Randles circuit with planar diffusion from Ji et al. [1]

    Notes
    -----

    .. math::


    **Parameters:**

    .. math::

        p[0] = R_{pore}; \\;
        p[1] = R_{ct,bulk}; \\;
        p[2] = C_{dl,bulk}; \\;
        p[3] = A_{w,bulk}; \\;
        p[4] = τ_{bulk}; \\;
        p[5] = R_{ct,surface}; \\;

    .. math::

        p[6] = C_{dl,surface}; \\;
        p[7] = N (\\text{number of circuit element}); \\;


    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    """

    N = int(p[7])
    frequencies = np.array(f)

    Rpore = p[0]/N
    Rct = p[1]*N
    Cdl = p[2]/N
    Aw = p[3]*N
    τd = p[4]
    Rs = p[5]*N
    Cs = p[6]/N

    Z1b = RCD([Rct, Cdl, Aw, τd], frequencies)

    Z1s = RC([Rs, Cs], frequencies)
    Zran = Z1b + Z1s
    Req = np.copy(Zran)
    inv_Zran = 1 / Zran
    for _ in range(1, N):

        Req = 1 / ((1 / (Req + Rpore)) + inv_Zran)

    return (Req)


@element(num_params=11, units=['Ohm', 'Ohm', 'F', 'Ohm', 's', 'Ohm', 'F', '-',
                               '1/V', '-', '-'])
def TLMDn(p, f):
    """

    2nd-NLEIS:
    Second harmonic nonlinear discrete transmission line model
    built based on the Randles circuit
    with planar diffusion from Ji et al. [1]

    Notes
    -----

    .. math::




    **Parameters:**

    .. math::
        p[0] = R_{pore}; \\;
        p[1] = R_{ct,bulk}; \\;
        p[2] = C_{dl,bulk}; \\;
        p[3] = A_{w,bulk}; \\;
        p[4] = τ_{bulk}; \\;
        p[5] = R_{ct,surface}; \\;

    .. math::
        p[6] = C_{dl,surface}; \\;
        p[7] = N (\\text{number of circuit element})
        p[8] = κ_{bulk}; \\;

    .. math::
        p[9] = ε_{bulk}; \\;
        p[10] = ε_{surface}; \\;



    """

    I1 = mTiD(p[0:8], f)  # calculate the current fraction (1st harmonic)

    N = int(p[7])
    frequencies = np.array(f)

    Rpore = p[0]/N
    Rct = p[1]*N
    Cdl = p[2]/N
    Aw = p[3]*N
    τd = p[4]
    Rs = p[5]*N
    Cs = p[6]/N
    κ = p[8]
    εb = p[9]
    εs = p[10]

    Z1b = RCD([Rct, Cdl, Aw, τd], frequencies)
    Z1s = RC([Rs, Cs], frequencies)
    Z2b = RCDn([Rct, Cdl, Aw, τd, κ, εb], frequencies)
    Z2s = RCn([Rs, Cs, εs], frequencies)
    Z1b2t = RCD([Rct, Cdl, Aw, τd], 2*frequencies)
    Z1s2t = RC([Rs, Cs], 2*frequencies)

    Z1 = Z1b + Z1s
    Z2 = Z2b + Z2s
    Z12t = Z1b2t + Z1s2t

    if N == 1:
        return (Z2)

    if N == 2:
        sum1 = Z1**2 / (2*Z1+Rpore)**2
        sum2 = (Z12t*Rpore+Rpore**2) / ((2*Z12t+Rpore)*(2*Z1+Rpore))
        Z = (sum1+sum2)*Z2
        return (Z)
    len_freq = len(frequencies)
    Z = np.zeros((len_freq), dtype=np.complex128)
    for freq_idx in range(len_freq):
        Ii = I1[freq_idx]
        Z12t_freq = Z12t[freq_idx]
        Z2_freq = Z2[freq_idx]
        rhs = np.zeros(N, dtype=np.complex128)
        rhs[:-1] = (Ii[:-1]**2 - Ii[-1]**2) * Z2_freq
        I2 = solve_tlm_second_harmonic_current(
            N, Rpore, Z12t_freq, rhs
        )
        Z[freq_idx] = Z2_freq*Ii[0]**2 + I2[-1]*Z12t_freq
    return (Z)


@element(num_params=8, units=['Ohm', 'Ohm', 'F', 'Ohm', 's', 'Ohm', 'F', '-'])
def mTiD(p, f):
    """

    EIS: current distribution of discrete transmission line model
    built based on the Randles circuit with planar diffusion from Ji et al. [1]

    Notes
    -----

    .. math::




    **Parameters:**

    .. math::
        p[0] = R_{pore}; \\;
        p[1] = R_{ct,bulk}; \\;
        p[2] = C_{dl,bulk}; \\;
        p[3] = A_{w,bulk}; \\;
        p[4] = τ_{bulk}; \\;
        p[5] = R_{ct,surface}; \\;

    .. math::
        p[6] = C_{dl,surface}; \\;
        p[7] = N (\\text{number of circuit element}); \\;


    [1] Y. Ji, D.T. Schwartz,
    Second-Harmonic Nonlinear Electrochemical Impedance Spectroscopy:
    I. Analytical theory and equivalent circuit representations
    for planar and porous electrodes.
    J. Electrochem. Soc. (2023). `doi: 10.1149/1945-7111/ad15ca
    <https://doi.org/10.1149/1945-7111/ad15ca>`_.

    """

    N = int(p[7])
    frequencies = f

    Rpore = p[0]/N
    Rct = p[1]*N
    Cdl = p[2]/N
    Aw = p[3]*N
    τd = p[4]
    Rs = p[5]*N
    Cs = p[6]/N

    Z1b = RCD([Rct, Cdl, Aw, τd], frequencies)

    Z1s = RC([Rs, Cs], frequencies)
    Zran = Z1b + Z1s
    Req = np.copy(Zran)
    inv_Zran = 1 / Zran
    for _ in range(1, N):

        Req = 1 / ((1 / (Req + Rpore)) + inv_Zran)

    Req = Req+Rpore

    return solve_tlm_current_distribution(Zran, Rpore, Req, N)


@element(num_params=11, units=['Ohm', 'Ohm', 'F', 'Ohm', 's', 'Ohm', 'F', '-',
                               '1/V', '-', '-'])
def mTiDn(p, f):
    """

    2nd-NLEIS: current distribution of
    Second harmonic nonlinear discrete transmission line model
    built based on the Randles circuit
    with planar diffusion from Ji et al. [1]

    Notes
    -----

    .. math::



    **Parameters:**

    .. math::

        p[0] = R_{pore}; \\;
        p[1] = R_{ct,bulk}; \\;
        p[2] = C_{dl,bulk}; \\;
        p[3] = A_{w,bulk}; \\;
        p[4] = τ_{bulk}; \\;
        p[5] = R_{ct,surface}; \\;


    .. math::

        p[6] = C_{dl,surface}; \\;
        p[7] = N (\\text{number of circuit element}); \\;
        p[8] = κ_{bulk}; \\;
        p[9] = ε_{bulk}; \\;
        p[10] = ε_{surface}; \\;



    """

    I1 = mTiD(p[0:8], f)  # calculate the current fraction (1st harmonic)

    N = int(p[7])
    frequencies = np.array(f)

    Rpore = p[0]/N
    Rct = p[1]*N
    Cdl = p[2]/N
    Aw = p[3]*N
    τd = p[4]
    Rs = p[5]*N
    Cs = p[6]/N
    κ = p[8]
    εb = p[9]
    εs = p[10]

    Z2b = RCDn([Rct, Cdl, Aw, τd, κ, εb], frequencies)
    Z2s = RCn([Rs, Cs, εs], frequencies)
    Z1b2t = RCD([Rct, Cdl, Aw, τd], 2*frequencies)
    Z1s2t = RC([Rs, Cs], 2*frequencies)

    Z2 = Z2b + Z2s
    Z12t = Z1b2t + Z1s2t

    if N == 1:
        return (0)
    len_freq = len(frequencies)
    I2 = np.zeros((len_freq, N), dtype=np.complex128)
    if N == 2:

        I2[:, 0] = Z2*Rpore / (2*Z12t+Rpore)**2
        I2[:, 1] = -Z2*Rpore / (2*Z12t+Rpore)**2

        return (I2)
    for freq_idx in range(len_freq):
        Ii = I1[freq_idx]
        rhs = np.zeros(N, dtype=np.complex128)
        rhs[:-1] = (Ii[:-1]**2 - Ii[-1]**2) * Z2[freq_idx]
        # Reverse the order to display the correct result from small to
        # larger N, matching the historical dense implementation.
        I2[freq_idx, :] = solve_tlm_second_harmonic_current(
            N, Rpore, Z12t[freq_idx], rhs
        )[::-1]

    return (I2)


@element(num_params=5, units=['Ohms', 'F', '-', 'Ohms', 's'])
def RCSQ(p, f):
    '''
    Beta element with CPE implementation
    EIS: Randles circuit (CPE element) with spherical diffusion

    Notes
    -----

    .. math::

    p[0] = Rct
    p[1] = Qdl
    p[2] = α
    p[3] = Aw
    p[4] = τd

    '''
    ω = 2 * np.pi * np.array(f)
    Rct, Qdl, alpha, Aw, τd = p[0], p[1], p[2], p[3], p[4]

    sqrt_1j_ω_τd = np.sqrt(1j * ω * τd)
    tanh_1j_ω_τd = np.tanh(sqrt_1j_ω_τd)
    Zd = Aw * tanh_1j_ω_τd / (sqrt_1j_ω_τd - tanh_1j_ω_τd)

    tau = Rct * Qdl
    Z = Rct / (Rct / (Rct + Zd) + tau * (1j * ω) ** alpha)
    return Z


@element(num_params=7, units=['Ohms', 'F', '-', 'Ohms', 's', '1/V', '-'])
def RCSQn(p, f):
    '''
    Beta element with CPE implementation
    2nd-NLEIS: Randles circuit (CPE element) with spherical diffusion

    Notes
    -----

    .. math::

    p[0] = Rct
    p[1] = Qdl
    p[2] = alpha
    p[3] = Aw
    p[4] = τd
    p[5] = κ
    p[6] = ε
    '''

    ω = 2 * np.pi * np.array(f)
    Rct, Qdl, alpha, Aw, τd, κ, ε = p[0], p[1], p[2], p[3], p[4], p[5], p[6]

    sqrt_1j_ω_τd = np.sqrt(1j * ω * τd)
    sqrt_1j_2ω_τd = np.sqrt(1j * 2 * ω * τd)
    tanh_1j_ω_τd = np.tanh(sqrt_1j_ω_τd)
    tanh_1j_2ω_τd = np.tanh(sqrt_1j_2ω_τd)

    Zd1 = Aw * tanh_1j_ω_τd / (sqrt_1j_ω_τd - tanh_1j_ω_τd)
    Zd2 = Aw * tanh_1j_2ω_τd / (sqrt_1j_2ω_τd - tanh_1j_2ω_τd)

    tau = Rct * Qdl
    y1 = Rct / (Zd1 + Rct)
    y2 = Zd1 / (Zd1 + Rct)

    Z1 = Rct / (y1 + tau * (1j * ω) ** alpha)
    const = ((Rct * κ * y2**2) - Rct * ε * F / (R * T) * y1**2) / (Zd2 + Rct)

    Z2 = (const * Z1**2) / (tau * (1j * 2 * ω) ** alpha + Rct / (Zd2 + Rct))

    return 0.5 * Z2


@element(num_params=5, units=['Ohms', 'F', '-', 'Ohms', 's'])
def RCDQ(p, f):
    '''
    Beta element with CPE implementation
    EIS: Randles circuit (CPE element) with spherical diffusion

    Notes
    -----

    .. math::

    p[0] = Rct
    p[1] = Qdl
    p[2] = alpha
    p[3] = Aw
    p[4] = τd

    '''
    ω = 2 * np.pi * np.array(f)
    Rct, Qdl, alpha, Aw, τd = p[0], p[1], p[2], p[3], p[4]

    sqrt_1j_ω_τd = np.sqrt(1j * ω * τd)
    tanh_1j_ω_τd = np.tanh(sqrt_1j_ω_τd)
    Zd = Aw / (sqrt_1j_ω_τd * tanh_1j_ω_τd)

    tau = Rct * Qdl
    Z = Rct / (Rct / (Rct + Zd) + tau * (1j * ω) ** alpha)
    return Z


@element(num_params=7, units=['Ohms', 'F', '-', 'Ohms', 's', '1/V', '-'])
def RCDQn(p, f):
    '''
    Beta element with CPE implementation
    2nd-NLEIS: Randles circuit (CPE element) with spherical diffusion

    Notes
    -----

    .. math::

    p[0] = Rct
    p[1] = Qdl
    p[2] = alpha
    p[3] = Aw
    p[4] = τd
    p[5] = κ
    p[6] = ε
    '''

    ω = 2 * np.pi * np.array(f)
    Rct, Qdl, alpha, Aw, τd, κ, ε = p[0], p[1], p[2], p[3], p[4], p[5], p[6]

    sqrt_1j_ω_τd = np.sqrt(1j * ω * τd)
    sqrt_1j_2ω_τd = np.sqrt(1j * 2 * ω * τd)
    tanh_1j_ω_τd = np.tanh(sqrt_1j_ω_τd)
    tanh_1j_2ω_τd = np.tanh(sqrt_1j_2ω_τd)

    Zd1 = Aw / (sqrt_1j_ω_τd * tanh_1j_ω_τd)
    Zd2 = Aw / (sqrt_1j_2ω_τd * tanh_1j_2ω_τd)

    tau = Rct * Qdl
    y1 = Rct / (Zd1 + Rct)
    y2 = Zd1 / (Zd1 + Rct)

    Z1 = Rct / (y1 + tau * (1j * ω) ** alpha)
    const = ((Rct * κ * y2**2) - Rct * ε * F / (R * T) * y1**2) / (Zd2 + Rct)

    Z2 = (const * Z1**2) / (tau * (1j * 2 * ω) ** alpha + Rct / (Zd2 + Rct))

    return 0.5 * Z2


@element(num_params=2, units=['Ohm', 's'])
def Kn(p, f):
    omega = 2 * np.pi * np.array(f)
    R2, tau_k = p[0], p[1]
    Z = R2 / (1 + 4*1j * omega * tau_k - 5 *
              (omega * tau_k)**2 - 2*1j*(omega * tau_k)**3)
    return (Z)


def get_element_from_name(name):
    excluded_chars = '0123456789_'
    return ''.join(char for char in name if char not in excluded_chars)


def typeChecker(p, f, name, length):
    assert isinstance(p, list), \
        'in {}, input must be of type list'.format(name)
    for i in p:
        assert isinstance(i, (float, int, np.int32, np.float64)), \
            'in {}, value {} in {} is not a number'.format(name, i, p)
    for i in f:
        assert isinstance(i, (float, int, np.int32, np.float64)), \
            'in {}, value {} in {} is not a number'.format(name, i, f)
    assert len(p) == length, \
        'in {}, input list must be length {}'.format(name, length)
    return
