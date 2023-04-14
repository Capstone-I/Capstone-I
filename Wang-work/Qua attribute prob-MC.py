import networkx as nx
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.algorithms import QAOA
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.library import TwoLocal

#It is from the Max_cut tutorial but significant change has been made based on my understanding and there are still some problems
service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q-asu/main/pi-deluca',
)
backend = service.backend('ibmq_qasm_simulator')

n_nodes = 4
edges = [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
G = nx.Graph()
G.add_weighted_edges_from(edges)

#attribute problem
problem = QuadraticProgram()
_ = [problem.binary_var(f'x_{i}') for i in range(n_nodes)]
quadratic = {}
for u, v, w in edges:
    quadratic[(f'x_{u}', f'x_{v}')] = w

problem.maximize(quadratic=quadratic)
converter = QuadraticProgramToQubo()
qubo = converter.convert(problem)

#num_binary_vars
ansatz = TwoLocal(qubo.num_binary_vars, ['ry', 'rz'], 'cx', reps=3, entanglement='linear', skip_final_rotation_layer=True)

qaoa = QAOA(ansatz=ansatz, quantum_instance=QuantumInstance(Aer.get_backend('aer_simulator'), shots=1024))
optimizer = MinimumEigenOptimizer(qaoa)
result = optimizer.solve(qubo)
original_solution = converter.interpret(result)

print("Max-Cut objective value:", original_solution.objective_value)
print("Max-Cut solution:", original_solution.x)
colors = ['r' if original_solution.x[i] == 0 else 'b' for i in range(n_nodes)]
nx.draw(G, node_color=colors, with_labels=True, font_weight='bold')
