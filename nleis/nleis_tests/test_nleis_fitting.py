import numpy as np

from impedance.preprocessing import ignoreBelowX

from impedance.tests.test_preprocessing import frequencies \
   as example_frequencies
from impedance.tests.test_preprocessing import Z_correct

from nleis.fitting import buildCircuit, \
   circuit_fit, mape,mae, extract_circuit_elements, \
   set_default_bounds, seq_fit_parm
from nleis.nleis_fitting import data_processing, \
simul_fit,individual_parameters
import os
test_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(test_dir, '../data')

def test_data_processing():

    # # get example data
    # # The example is shown in "Getting Started" page 
    
    frequencies = np.loadtxt(os.path.join(data_dir, 'freq_30a.txt'))
    Z1 = np.loadtxt(os.path.join(data_dir,'Z1s_30a.txt')).view(complex)[1]
    Z2 = np.loadtxt(os.path.join(data_dir,'Z2s_30a.txt')).view(complex)[1]

    max_f = 10

    f,Z1,Z2,f2_truncated,Z2_truncated = data_processing(frequencies,Z1,Z2,max_f=max_f)
    assert len(f)==len(Z1)==len(Z2)
    assert Z1.imag.max() < 0 
    assert f2_truncated.max()<max_f
    assert len(f2_truncated) == len(Z2_truncated)



def test_set_default_bounds():
    # Test example circuit that calculate the difference between
    # two porous electrode with spherical particle and with platelet like particle 
    circuit = 'd(TDSn0,TDPn1)'

    # Test with no constants
    default_bounds = (np.zeros(14), np.inf*np.ones(14))
    default_bounds[0][5] = -np.inf
    default_bounds[0][6] = -0.5
    default_bounds[1][6] = 0.5
    default_bounds[0][12] = -np.inf
    default_bounds[0][13] = -0.5
    default_bounds[1][13] = 0.5

    bounds_from_func = set_default_bounds(circuit)

    assert np.allclose(default_bounds, bounds_from_func)

    # Test with constants
    constants = {'TDSn0_0': 1}
    default_bounds = (np.zeros(13), np.inf*np.ones(13))
    default_bounds[0][4] = -np.inf
    default_bounds[0][5] = -0.5
    default_bounds[1][5] = 0.5
    default_bounds[0][11] = -np.inf
    default_bounds[0][12] = -0.5
    default_bounds[1][12] = 0.5
    bounds_from_func = set_default_bounds(circuit, constants=constants)

    assert np.allclose(default_bounds, bounds_from_func)

#### This test remain unchanged as we are adopting it from impedance.py
def test_circuit_fit():

    # Test trivial model (10 Ohm resistor)
    circuit = 'R0'
    initial_guess = [10]

    results_simple = [10]

    frequencies = np.array([10, 100, 1000])
    Z_data = np.array([10, 10, 10])  # impedance is real

    assert np.allclose(circuit_fit(frequencies, Z_data, circuit,
                                   initial_guess, constants={},
                                   global_opt=True)[0],
                       results_simple, rtol=1e-1)

    # check that list inputs work
    frequency_list = [10, 100, 1000]
    Z_data_list = [10, 10, 10]

    assert np.allclose(circuit_fit(frequency_list, Z_data_list, circuit,
                                   initial_guess, constants={},
                                   global_opt=True)[0],
                       results_simple, rtol=1e-1)

    # Test example circuit from "Getting Started" page
    circuit = 'R0-p(R1,C1)-p(R2-Wo1,C2)'
    initial_guess = [.01, .01, 100, .01, .05, 100, 1]
    bounds = [(0, 0, 0, 0, 0, 0, 0),
              (10, 1, 1e3, 1, 1, 1e4, 100)]

    # results change slightly using predefined bounds
    results_local = np.array([1.65e-2, 8.68e-3, 3.32, 5.39e-3,
                              6.31e-2, 2.33e2, 2.20e-1])
    results_local_bounds = results_local.copy()
    results_local_bounds[5] = 2.38e2
    results_local_weighted = np.array([1.64e-2, 9.06e-3, 3.06,
                                       5.29e-3, 1.45e-1, 1.32e3, 2.02e-1])

    results_global = np.array([1.65e-2, 5.34e-3, 0.22, 9.15e-3,
                               1.31e-1, 1.10e3, 2.78])

    # Filter
    example_frequencies_filtered, \
        Z_correct_filtered = ignoreBelowX(example_frequencies, Z_correct)

    # Test local fitting
    assert np.allclose(circuit_fit(example_frequencies_filtered,
                                   Z_correct_filtered, circuit,
                                   initial_guess, constants={})[0],
                       results_local, rtol=1e-2)

    # Test local fitting with predefined bounds
    assert np.allclose(circuit_fit(example_frequencies_filtered,
                                   Z_correct_filtered, circuit,
                                   initial_guess, bounds=bounds,
                                   constants={})[0],
                       results_local_bounds, rtol=1e-2)

    # Test local fitting with predefined weights
    # Use abs(Z), stacked in order of (Re, Im) components
    sigma = np.hstack((np.abs(Z_correct_filtered),
                       np.abs(Z_correct_filtered)))
    assert np.allclose(circuit_fit(example_frequencies_filtered,
                                   Z_correct_filtered, circuit,
                                   initial_guess, sigma=sigma,
                                   constants={})[0],
                       results_local_weighted, rtol=1e-2)

    # Test if using weight_by_modulus=True produces the same results
    assert np.allclose(circuit_fit(example_frequencies_filtered,
                                   Z_correct_filtered, circuit,
                                   initial_guess, weight_by_modulus=True,
                                   constants={})[0],
                       results_local_weighted, rtol=1e-2)

    # Test global fitting on multiple seeds
    # All seeds should converge to the same parameter values
    # seed = 0 (default)
    assert np.allclose(circuit_fit(example_frequencies_filtered,
                                   Z_correct_filtered, circuit,
                                   initial_guess, constants={},
                                   global_opt=True)[0],
                       results_global, rtol=1e-1)

    # seed = 0, with predefined bounds
    assert np.allclose(circuit_fit(example_frequencies_filtered,
                                   Z_correct_filtered, circuit,
                                   initial_guess, constants={},
                                   global_opt=True, bounds=bounds,
                                   seed=0)[0],
                       results_global, rtol=1e-1)

    # seed = 1
    assert np.allclose(circuit_fit(example_frequencies_filtered,
                                   Z_correct_filtered, circuit,
                                   initial_guess, constants={},
                                   global_opt=True, seed=1)[0],
                       results_global, rtol=1e-1)

    # seed = 42
    assert np.allclose(circuit_fit(example_frequencies_filtered,
                                   Z_correct_filtered, circuit,
                                   initial_guess, constants={},
                                   global_opt=True, seed=42)[0],
                       results_global, rtol=1e-1)
    

def test_buildCircuit():

   # Test a simple porous electrode with contact resistor and inductance
   circuit = 'L0-R0-TDS0'
   params = [1e-5, 0.01, 1, 0.1, 10, 1, 1000]
   frequencies = [1000.0, 5.0, 0.01]

   assert buildCircuit(circuit, frequencies, *params,
                       constants={})[0].replace(' ', '') == \
       's([L([1e-05],[1000.0,5.0,0.01]),' + \
       'R([0.01],[1000.0,5.0,0.01]),' + \
       'TDS([1.0,0.1,10.0,1.0,1000.0],[1000.0,5.0,0.01])])' 

# Test the substraction between two elements
   circuit = 'd(R1,R2)'
   params = [0.1, 0.2]
   frequencies = [1000.0, 5.0, 0.01]

   assert buildCircuit(circuit, frequencies, *params,
                       constants={})[0].replace(' ', '') == \
       'd([R([0.1],[1000.0,5.0,0.01]),' + \
       'R([0.2],[1000.0,5.0,0.01])])'

   # Test nested parallel and difference groups
   circuit = 'R0-d(p(R1, C1)-R2, C2)'
   params = [1, 2, 3, 4, 5]
   frequencies = [1000.0, 5.0, 0.01]

   assert buildCircuit(circuit, frequencies, *params,
                       constants={})[0].replace(' ', '') == \
       's([R([1],[1000.0,5.0,0.01]),' + \
       'd([s([p([R([2],[1000.0,5.0,0.01]),' + \
       'C([3],[1000.0,5.0,0.01])]),' + \
       'R([4],[1000.0,5.0,0.01])]),' + \
       'C([5],[1000.0,5.0,0.01])])])'


   # Test single element circuit
   circuit = 'R1'
   params = [100]
   frequencies = [1000.0, 5.0, 0.01]

   assert buildCircuit(circuit, frequencies, *params,
                       constants={})[0].replace(' ', '') == \
       'R([100],[1000.0,5.0,0.01])'


def test_mae():
   a = np.array([2 + 4*1j, 3 + 2*1j])
   b = np.array([2 + 4*1j, 3 + 2*1j])

   assert mae(a, b) == 0.0

   c = np.array([2 + 4*1j, 1 + 4*1j])
   d = np.array([4 + 2*1j, 3 + 2*1j])
   assert np.isclose(mae(c, d), np.sqrt(8))

def test_mape():
   a = np.array([2 + 4*1j, 3 + 2*1j])
   b = np.array([2 + 4*1j, 3 + 2*1j])

   assert mape(a, b) == 0.0

   c = np.array([2 + 4*1j, 1 + 4*1j])
   d = np.array([4 + 2*1j, 3 + 2*1j])
   assert np.isclose(mape(c, d), 0.5*np.sqrt(8)*(1/np.sqrt(20)+1/np.sqrt(17))*100)

def test_element_extraction():
   circuit = 'd(TDSn0,TDPn1)'
   extracted_elements = extract_circuit_elements(circuit)
   assert extracted_elements == ['TDSn0', 'TDPn1']

def test_seq_fit_parm():
   input_dic = {'TDS0_1':0,'TDS1_3':0}
   target_arr = ['TDS0','TDS1']
   output_arr = ['TDSn0','TDSn1']
   output_dic = seq_fit_parm(input_dic,target_arr,output_arr)
   assert output_dic == {'TDSn0_1':0,'TDSn1_3':0}


def test_simul_fit():
    # get example data
    # The example is shown in "Getting Started" page 
    
    frequencies = np.loadtxt(os.path.join(data_dir, 'freq_30a.txt'))
    Z1 = np.loadtxt(os.path.join(data_dir,'Z1s_30a.txt')).view(complex)[1]
    Z2 = np.loadtxt(os.path.join(data_dir,'Z2s_30a.txt')).view(complex)[1]

    circ_str_1 = 'L0-R0-TDS0-TDS1'
    circ_str_2 = 'd(TDSn0,TDSn1)'
    edited_circuit = 'L0-R0-TDSn0-TDSn1'

   #  initial_guess = [1e-7,1e-3 # L0,RO
   #                  ,5e-3,1e-3,10,1e-2,100,10,0.1 ## TDS0 + additioal nonlinear parameters
   #                  ,1e-3,1e-3,1e-3,1e-2,1000,0,0 ## TDS1 + additioal nonlinear parameters
   #                  ]
   ## The test framework here can be improved in the future after getting much stable initial guess
    results =[9.81376136e-08,1.34551860e-02,2.52398185e-02,5.06155989e-03,
         8.82298681,8.78424359e-05,3.58691184,1.22575311e+01,
         8.74768343e-02,2.09036506e-02,1.13813146e-03,8.13615357e-01,
         1.86194200e+02,3.29019412e+09,1.02283674,6.38865883e-03]
    initial_guess= results
    
    p, perror =  simul_fit(frequencies, Z1, Z2, circ_str_1,circ_str_2, edited_circuit, initial_guess, constants_1={},constants_2={},
                bounds = None, opt='max',cost = 0.5,max_f=10,param_norm = True,positive = True)

    assert np.allclose (p,results)

   
def test_individual_parameters():

    ## EIS and 2nd-NLEIS circ_str are defined here to make sure 
    ## we know we have a a pair of consistent models 

    circ_str_1 = 'L0-R0-TDS0-TDS1'
    circ_str_2 = 'd(TDSn0,TDSn1)'

    initial_guess = [1e-7,1e-3 # L0,RO
                    ,5e-3,1e-3,10,1e-2,100,10,0.1 ## TDS0 + additioal nonlinear parameters
                    ,1e-3,1e-3,1e-3,1e-2,1000,0,0 ## TDS1 + additioal nonlinear parameters
                    ]
    ## Test without constant
    p1, p2 = individual_parameters(circ_str_1,initial_guess,constants_1 = {},constants_2= {})
    assert np.allclose(p1,[1e-7,1e-3
                    ,5e-3,1e-3,10.0,1e-02,100.0
                    ,1e-3,1e-03,1e-03,1e-02,1000.0])
    assert np.allclose(p2,[5e-3,1e-3,10,1e-2,100,10,0.1
                    ,1e-3,1e-3,1e-3,1e-2,1000,0,0])
    
    ## Test with constant
    initial_guess = [1e-7,1e-3 # L0,RO
                ,5e-3,10,1e-2,100,10,0.1 ## TDS0 + additioal nonlinear parameters
                ,1e-3,1e-3,1e-3,1e-2,1000,0 ## TDS1 + additioal nonlinear parameters
                ]
    p1, p2 = individual_parameters(circ_str_1,initial_guess
                                   ,constants_1 = {'TDS0_1':0},constants_2= {'TDSn1_6':0})

    assert np.allclose(p1,[1e-7,1e-3
                    ,5e-3,10,1e-02,100.0
                    ,1e-3,1e-03,1e-03,1e-02,1000.0])
    assert np.allclose(p2,[5e-3,10,1e-2,100,10,0.1
                    ,1e-3,1e-3,1e-3,1e-2,1000,0])

