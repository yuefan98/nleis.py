import matplotlib.pyplot as plt
import numpy as np
from nleis.visualization import plot_altair, plot_first, plot_second
import json


def test_plot_nyquist():

    Z = np.array([1, 2, 3]) + 1j*np.array([2, 3, 4])

    # pass axes
    _, ax = plt.subplots()
    ax = plot_first(ax, Z, scale=10)
    xs, ys = ax.lines[0].get_xydata().T
    assert (xs == Z.real).all() and (ys == -Z.imag).all()

    
    # pass axes
    _, ax = plt.subplots()
    ax = plot_second(ax, Z, scale=10)
    xs, ys = ax.lines[0].get_xydata().T
    assert (xs == Z.real).all() and (ys == -Z.imag).all()



def test_plot_altair():
    frequencies = [1000.0, 1.0, 0.01]
    Z = np.array([1, 2, 3]) + 1j*np.array([2, 3, 4])

    chart = plot_altair({'data': {'f': frequencies, 'Z': Z},
                         'fit': {'f': frequencies, 'Z': Z, 'fmt': '-'}},
                        size=400)

    datasets = json.loads(chart.to_json())['datasets']
    for dataset in datasets.keys():
        assert len(datasets[dataset]) == len(Z)


