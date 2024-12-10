from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer.aerprovider import AerSimulator
from math import gcd
from fractions import Fraction
import numpy as np

class ShorsAlgorithm:
    def __init__(self, N, shots=1000):
        if not isinstance(N, int) or N <= 1:
            raise ValueError("N must be an integer greater than 1.")
        self.N = N
        if not isinstance(shots, int) or shots <= 0:
            raise ValueError("Number of shots must be a positive integer.")
        self.shots = shots
        
        # calculate the number of qubits required based on the binary representation of N
        self.n = len(bin(N)[2:])
        self.simulator = AerSimulator()  # set up the quantum simulator
        
    def _apply_qft_inverse(self, qc, qubits):
        # inverse QFT implementation
        for i in range(len(qubits)//2):
            qc.swap(qubits[i], qubits[-(i+1)])  # swap qubits symmetrically
        
        for j in range(len(qubits)):
            for k in range(j):
                # apply controlled-phase gates for inverse Fourier Transform
                qc.cp(-np.pi/float(2**(j-k)), qubits[k], qubits[j])
            qc.h(qubits[j])  # apply a Hadamard gate to the qubit
        
    def _quantum_period_finding(self, a):
        # perform the quantum part of Shor's algorithm to find the period
        n_count = 2 * self.n  # number of counting qubits
        n_power = self.n      # number of qubits for modular exponentiation
        
        # define quantum and classical registers
        q_count = QuantumRegister(n_count, 'count')  # counting qubits
        q_power = QuantumRegister(n_power, 'power')  # target qubits for modular exponentiation
        c_reg = ClassicalRegister(n_count, 'c')      # classical register to store measurement
        qc = QuantumCircuit(q_count, q_power, c_reg) # quantum circuit
        
        # initialize counting qubits in superposition
        for i in range(n_count):
            qc.h(q_count[i])
            
        qc.x(q_power[0])  # set |1> state in the power register
        
        # modular exponentiation controlled by counting qubits
        for i in range(n_count):
            power = pow(a, 2**i, self.N)  # calculate modular exponentiation a^(2^i) mod N
            self._controlled_modular_multiplication(qc, q_count[i], q_power, power)
        
        # apply inverse QFT to the counting register
        self._apply_qft_inverse(qc, q_count)
        
        # measure the counting register to extract the phase
        qc.measure(q_count, c_reg)
        
        # simulate the quantum circuit and get the results
        job = self.simulator.run(qc, shots=self.shots)
        result = job.result()
        counts = result.get_counts()  # get the measurement results
        
        return max(counts, key=counts.get)  # return the most frequent measurement result
    
    def _controlled_modular_multiplication(self, qc, control, target_reg, power):
        # perform controlled modular multiplication for a given power
        n = len(target_reg)  # number of qubits in the target register
        for i in range(n):
            # apply controlled phase rotation proportional to the power
            qc.cp(2 * np.pi * power / (2**(i+1)), control, target_reg[i])
    
    def _classical_post_processing(self, measurement):
        # convert quantum measurement to a classical phase
        phase = int(measurement, 2) / (2 ** (2 * self.n))  # convert binary measurement to a fraction
        frac = Fraction(phase).limit_denominator(self.N)   # find the closest rational approximation
        return frac.denominator  # the denominator corresponds to the period
    
    def factor(self):
        if self.N % 2 == 0:  # check if N is even
            return 2, self.N // 2
        
        a = np.random.randint(2, self.N)  # choose a random integer in [2, N-1]
        g = gcd(a, self.N)  # compute GCD(a, N)
        
        if g != 1:  # if a shares a common factor with N, we found a factor
            return g, self.N // g
        
        # use quantum period finding to determine the period
        measurement = self._quantum_period_finding(a)
        r = self._classical_post_processing(measurement)
        
        if r % 2 != 0:  # if the period is odd, retry
            return self.factor()
        
        # compute potential factors using the period
        x = pow(a, r//2, self.N)
        factor1 = gcd(x + 1, self.N)
        factor2 = gcd(x - 1, self.N)
        
        if factor1 * factor2 != self.N:  # retry if factors are incorrect
            return self.factor()
            
        return factor1, factor2  # return the found factors

def factor_number(N, shots):
    shor = ShorsAlgorithm(N, shots)
    return shor.factor()

if __name__ == "__main__":
    N = int(input("Enter a number to factor: "))
    shots = int(input("Enter the number of times the quantum circuit must be executed and measured: "))
    print(f"Finding factors of {N} with {shots} iterations...")
    factors = factor_number(N, shots)
    print(f"The factors of {N} are: {factors}")
