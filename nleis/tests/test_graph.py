import numpy as np
import matplotlib.pyplot as plt  # noqa: F401

import timeit

from nleis.nleis import EISandNLEIS, NLEISCustomCircuit  # noqa: F401
from nleis.fitting import CircuitGraph


def test_graph_NLEISCustomCircuit():
    circ_str = 'd(TDSn0,TDSn1)'
    initial_guess = [
        5e-3, 1e-3, 10, 1e-2, 100, 10, 0.1,
        # TDS0 + additioal nonlinear parameters
        1e-3, 1e-3, 1e-3, 1e-2, 1000, 1, 0.01,
        # TDS1 + additioal nonlinear parameters
    ]
    # initialize
    eval_circuit = NLEISCustomCircuit(
        circ_str, initial_guess=initial_guess, graph=False)
    graph_circuit = NLEISCustomCircuit(
        circ_str, initial_guess=initial_guess, graph=True)

    frequencies = np.geomspace(1e-3, 1e1, 100)

    def time_predict_eval():
        return eval_circuit.predict(frequencies, max_f=np.inf)

    def time_predict_graph():
        return graph_circuit.predict(frequencies, max_f=np.inf)
    eval_time = timeit.timeit(time_predict_eval, number=20)
    graph_time = timeit.timeit(time_predict_graph, number=20)
    Z2_eval = eval_circuit.predict(frequencies, max_f=np.inf)
    Z2_graph = graph_circuit.predict(frequencies, max_f=np.inf)
    assert np.allclose(Z2_eval, Z2_graph)
    assert graph_time < eval_time
    print(f'eval_time: {eval_time}, graph_time: {graph_time}')

    # test the fitting of the graph circuit using curve_fit
    graph_circuit.fit(frequencies, Z2_graph, max_f=np.inf)
    assert np.allclose(graph_circuit.predict(
        frequencies, max_f=np.inf), Z2_graph)

    # test the fitting of the graph circuit using with global_opt = True
    graph_circuit.fit(frequencies, Z2_graph, max_f=np.inf, global_opt=True)
    assert np.allclose(graph_circuit.predict(
        frequencies, max_f=np.inf), Z2_graph)


def test_graph_EISandNLEIS():
    circ_str_1 = 'L0-R0-TDS0-TDS1'
    circ_str_2 = 'd(TDSn0,TDSn1)'
    initial_guess = [1e-7, 1e-3,  # L0,RO
                     5e-3, 1e-3, 10, 1e-2, 100, 10, 0.1,
                     # TDS0 + additioal nonlinear parameters
                     1e-3, 1e-3, 1e-3, 1e-2, 1000, 1, 0.01,
                     # TDS1 + additioal nonlinear parameters
                     ]

    eval_circuit = EISandNLEIS(
        circ_str_1, circ_str_2, initial_guess=initial_guess, graph=False)
    graph_circuit = EISandNLEIS(circ_str_1, circ_str_2,
                                initial_guess=initial_guess, graph=True)

    frequencies = np.geomspace(1e-3, 1e3, 100)

    def time_predict_eval():
        return eval_circuit.predict(frequencies)

    def time_predict_graph():
        return graph_circuit.predict(frequencies)
    eval_time = timeit.timeit(time_predict_eval, number=20)
    graph_time = timeit.timeit(time_predict_graph, number=20)

    Z1_eval, Z2_eval = eval_circuit.predict(frequencies, max_f=np.inf)
    Z1_graph, Z2_graph = graph_circuit.predict(frequencies, max_f=np.inf)
    assert np.allclose(Z1_eval, Z1_graph)
    assert np.allclose(Z2_eval, Z2_graph)

    assert graph_time < eval_time
    print(f'eval_time: {eval_time}, graph_time: {graph_time}')

    # test the fitting of the graph circuit using curve_fit and opt = 'max'
    graph_circuit.fit(frequencies, Z1_graph, Z2_graph,  max_f=np.inf)
    assert np.allclose(graph_circuit.predict(
        frequencies, max_f=np.inf), (Z1_graph, Z2_graph))

    # test the fitting of the graph circuit using curve_fit and opt = 'neg'
    graph_circuit.fit(frequencies, Z1_graph, Z2_graph, opt='neg', max_f=np.inf)
    assert np.allclose(graph_circuit.predict(
        frequencies, max_f=np.inf), (Z1_graph, Z2_graph))


def test_CircuitGraph():
    # Test multiple parallel elements
    circuit = "R0-p(C1,R1,R2)"

    assert len(CircuitGraph(circuit).execution_order) == 6

    # Test nested parallel groups
    circuit = "R0-p(p(R1, C1)-R2, C2)"

    assert len(CircuitGraph(circuit).execution_order) == 9

    # Test parallel elements at beginning and end
    circuit = "p(C1,R1)-p(C2,R2)"

    assert len(CircuitGraph(circuit).execution_order) == 7

    # Test single element circuit
    circuit = "R1"

    assert len(CircuitGraph(circuit).execution_order) == 1

    # Test complex circuit with d and p
    circuit = "d(p(R1,C1),p(R2,C2))"
    params = [0.1, 0.01, 1, 1000]
    frequencies = [1000.0, 5.0, 0.01]

    cg = CircuitGraph(circuit)

    # test for execution_order
    assert len(cg.execution_order) == 7

    # test for compute
    assert len(cg.compute(frequencies, *params)) == len(frequencies)

    # test for calculate_circuit_length
    assert cg.calculate_circuit_length() == 4

    # test for __call__
    assert np.allclose(cg.compute(
        frequencies, *params), cg(frequencies, *params))

    # test for __eq__
    assert not cg == 1
    cg2 = CircuitGraph(circuit)
    assert cg == cg2

    # test for graph visualization
    f, ax = plt.subplots()
    cg.visualize_graph(ax=ax)
    plt.close(f)

    # test for constants inputs
    circuit = "d(p(R1,C1),p(R2,C2))"
    params1 = [0.1, 0.01, 1]
    constants = {'C2': 1000}

    cg = CircuitGraph(circuit, constants)

    assert np.allclose(cg(frequencies, *params1), cg2(frequencies, *params))
