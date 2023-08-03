from qiskit import QuantumCircuit, transpile, assemble, Aer, execute
import numpy as np

def create_random_number_circuit(bits):
    circuit = QuantumCircuit(bits, bits)
    
    for i in range(bits):
        circuit.h(i)
        
    return circuit

num_bits = 5

random_circuit = create_random_number_circuit(num_bits)

simulator = Aer.get_backend('qasm_simulator')
job = execute(random_circuit, simulator, shots=1)
result = job.result()
counts = result.get_counts()

random_number = int(list(counts.keys())[0], 2)
print("Random number: ", random_number)
