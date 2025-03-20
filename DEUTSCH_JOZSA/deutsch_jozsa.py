from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer.aerprovider import AerSimulator
import numpy as np
import random

class DeutschJozsaAlgorithm:
    def __init__(self, n_qubits, oracle_type='balanced', constant_value=0, custom_bitstring=None, shots=1024):
        if not isinstance(n_qubits, int) or n_qubits <= 0:
            raise ValueError("N must be a positive integer.")
        self.n_qubits = n_qubits
        
        valid_oracle_types = ['constant', 'balanced', 'custom']
        if oracle_type not in valid_oracle_types:
            raise ValueError(f"Oracle type must be one of: {', '.join(valid_oracle_types)}")
        self.oracle_type = oracle_type
        
        if constant_value not in [0, 1]:
            raise ValueError("Constant value must be 0 or 1.")
        self.constant_value = constant_value
        
        self.custom_bitstring = custom_bitstring
        if oracle_type == 'custom' and custom_bitstring is None:
            self._generate_random_balanced_oracle()
        elif oracle_type == 'custom' and custom_bitstring is not None:
            if len(custom_bitstring) != 2**n_qubits:
                raise ValueError(f"Custom bitstring length must be 2^{n_qubits} = {2**n_qubits}")
            if not all(bit in '01' for bit in custom_bitstring):
                raise ValueError("Custom bitstring must contain only 0 and 1")
        
        if not isinstance(shots, int) or shots <= 0:
            raise ValueError("Number of shots must be a positive integer.")
        self.shots = shots
        
        self.simulator = AerSimulator()
    
    def _generate_random_balanced_oracle(self):
        total_inputs = 2**self.n_qubits
        balanced_bits = ['0'] * (total_inputs // 2) + ['1'] * (total_inputs // 2) # create a list with half 0s and half 1s
        random.shuffle(balanced_bits) # then shuffle it
        self.custom_bitstring = ''.join(balanced_bits)
        
    def _create_oracle(self, qc, qubits, target_qubit):
        if self.oracle_type == 'constant':
            # oracle for constant function: f(x) = 0 or f(x) = 1
            if self.constant_value == 1:
                # for f(x) = 1, apply X to the target (flipping)
                qc.x(target_qubit)
                
        elif self.oracle_type == 'balanced':
            # balanced function using a series of quantum gates that create the function:
            # f(x) = x₁ ⊕ x₂ ⊕ ... ⊕ xₙ
            
            # implement controlled-NOT from the first qubit to the target
            qc.cx(qubits[0], target_qubit)
            
            # for each other qubit apply another controlled-NOT to the target
            for i in range(1, len(qubits)):
                qc.cx(qubits[i], target_qubit)
        
        elif self.oracle_type == 'custom':
            
            n = len(qubits)
            
            # iterate through all possible input combinations
            for i in range(2**n):
                # convert index i to binary string and add leading zeros
                binary = format(i, f'0{n}b')
                
                # check if this input should produce output 1
                if self.custom_bitstring[i] == '1':
                    # add X gates to prepare the base state
                    for j in range(n):
                        if binary[j] == '0':
                            qc.x(qubits[j])
                    
                    # add a multi-controlled-X gate
                    if n == 1:
                        qc.cx(qubits[0], target_qubit)
                    else:
                        # for n>1, use a multi-control approach
                        qc.mcx(qubits, target_qubit)
                    
                    # restore input qubits to their original state
                    for j in range(n):
                        if binary[j] == '0':
                            qc.x(qubits[j])
    
    def run(self):
        # define quantum and classical registers
        input_register = QuantumRegister(self.n_qubits, 'input') 
        output_register = QuantumRegister(1, 'output') 
        classical_register = ClassicalRegister(self.n_qubits, 'c')
        
        qc = QuantumCircuit(input_register, output_register, classical_register)
        qc.x(output_register[0]) # initialize the output register to |1>
        
        # apply Hadamard to all qubits
        for i in range(self.n_qubits):
            qc.h(input_register[i])
        qc.h(output_register[0])
        
        self._create_oracle(qc, input_register, output_register[0])
        
        # apply Hadamard to the input qubits
        for i in range(self.n_qubits):
            qc.h(input_register[i])
                
        qc.measure(input_register, classical_register) # measure the input qubits
        
        job = self.simulator.run(qc, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        # interpret the results:
        # if the function is constant, all input qubits will be |0> after measurement
        # if the function is balanced, at least one qubit will be |1>
        if '0' * self.n_qubits in counts and counts['0' * self.n_qubits] > self.shots / 2:
            return 'constant'
        else:
            return 'balanced'
    
    def get_circuit(self):
        # create quantum registers
        input_register = QuantumRegister(self.n_qubits, 'input')
        output_register = QuantumRegister(1, 'output')
        classical_register = ClassicalRegister(self.n_qubits, 'c')
        
        qc = QuantumCircuit(input_register, output_register, classical_register)
        
        # initialize the output register to |1>
        qc.x(output_register[0])
        
        # apply Hadamard to all qubits
        for i in range(self.n_qubits):
            qc.h(input_register[i])
        qc.h(output_register[0])
        
        self._create_oracle(qc, input_register, output_register[0])
        
        for i in range(self.n_qubits):
            qc.h(input_register[i])
        
        qc.measure(input_register, classical_register)
        
        return qc
    
    def validate_oracle(self):
        if self.oracle_type == 'constant':
            return 'constant'
        
        if self.oracle_type == 'balanced':
            return 'balanced'
        
        if self.oracle_type == 'custom':
            ones_count = self.custom_bitstring.count('1') # count how many 1s are in the custom bitstring
            total = len(self.custom_bitstring)
            
            if ones_count == 0 or ones_count == total:
                return 'constant'
            elif ones_count == total // 2:
                return 'balanced'
            else:
                return f'neither (ones: {ones_count}, zeros: {total-ones_count})'

def run_deutsch_jozsa(n_qubits, oracle_type='balanced', constant_value=0, custom_bitstring=None, shots=1024):
    dj = DeutschJozsaAlgorithm(n_qubits, oracle_type, constant_value, custom_bitstring, shots)
    return dj.run()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        n_qubits = int(sys.argv[1])
        oracle_type = sys.argv[2] if len(sys.argv) > 2 else 'balanced'
        constant_value = int(sys.argv[3]) if len(sys.argv) > 3 and oracle_type == 'constant' else 0
        shots = int(sys.argv[-1]) if len(sys.argv) > 4 else 1024
        
        dj = DeutschJozsaAlgorithm(n_qubits, oracle_type, constant_value, None, shots)
    else:
        n_qubits = int(input("Enter the number of qubits: "))
        
        print("Available oracle types:")
        print("1. Constant (all 0s or all 1s)")
        print("2. Balanced (half 0s, half 1s)")
        print("3. Custom (user-defined)")
        oracle_choice = int(input("Choose oracle type (1-3): "))
        
        if oracle_choice == 1:
            oracle_type = 'constant'
            constant_value = int(input("Enter constant value (0 or 1): "))
            custom_bitstring = None
        elif oracle_choice == 2:
            oracle_type = 'balanced'
            constant_value = 0
            custom_bitstring = None
        else:
            oracle_type = 'custom'
            constant_value = 0
            print(f"You need to specify a string of {2**n_qubits} bits (0s and 1s):")
            print("1. Enter manually")
            print("2. Generate a random balanced string")
            bitstring_choice = int(input("Choose (1-2): "))
            
            if bitstring_choice == 1:
                custom_bitstring = input(f"Enter {2**n_qubits} bits (0s and 1s only): ")
            else:
                custom_bitstring = None 
                
        shots = int(input("Enter the number of executions: "))
        
        dj = DeutschJozsaAlgorithm(n_qubits, oracle_type, constant_value, custom_bitstring, shots)
    
    actual_type = dj.validate_oracle()
    print(f"Configured oracle: {oracle_type}")
    print(f"Actual oracle behavior: {actual_type}")
    
    print(f"\nRunning Deutsch-Jozsa algorithm with {n_qubits} qubits, {shots} shots...")
    result = dj.run()
    print(f"The oracle function was classified as: {result}")
    
    circuit = dj.get_circuit()
    print("\nQuantum circuit:")
    print(circuit) 