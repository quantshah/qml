r"""
.. _grant:

An initialization strategy to address barren plateaus in quantum neural networks
================================================================================
*Author: Shahnawaz Ahmed (shahnawaz.ahmed95@gmail.com)*

A random variational quantum circuit suffers from the problem of barren plateaus
as we see in this `PennyLane tutorial
<https://pennylane.readthedocs.io/en/latest/tutorials/pennylane_run_barren_plateaus.html#barren_plateaus>`_.

One way to deal with such barren plateaus is proposed as
*"randomly selecting some of the initial
  parameter values, then choosing the remaining values so that
  the final circuit is a sequence of shallow unitary blocks that
  each evaluates to the identity. Initializing in this way limits
  the effective depth of the circuits used to calculate the first
  parameter update so that they cannot be stuck in a barren plateau
  at the start of training."*

In this tutorial we will make a simple quantum circuit implementing this
initialization strategy.

"""

##############################################################################
# Exploring the barren plateau problem with PennyLane
# ---------------------------------------------------
#
# First, we import PennyLane, NumPy, and Matplotlib

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt


##################################################
# Next, we create a randomized variational circuit

# Set a seed for reproducibility
np.random.seed(42)


def rand_circuit(params, random_gate_sequence=None, num_qubits=None):
    """A random variational quantum circuit.

    Args:
        params (array[float]): array of parameters
        random_gate_sequence (dict): a dictionary of random gates
        num_qubits (int): the number of qubits in the circuit

    Returns:
        float: the expectation value of the target observable
    """
    for i in range(num_qubits):
        random_gate_sequence[i](params[0, i], wires=i)

    for i in range(num_qubits):
        random_gate_sequence[i](params[1, i], wires=i)

    H = np.zeros((2 ** num_qubits, 2 ** num_qubits))
    H[0, 0] = 1
    wirelist = [i for i in range(num_qubits)]
    return qml.expval(qml.Hermitian(H, wirelist))


############################################################
# Now we can compute the gradient and calculate the variance.
# While we only sample 200 random circuits to allow the code
# to run in a reasonable amount of time, this can be
# increased for more accurate results. We only consider the
# gradient of the output wr.t. to the last parameter in the
# circuit. Hence we choose to save gradient[-1] only.

num_samples = 200

###########################################################
# Evaluate the gradient for more qubits
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can repeat the above analysis with increasing number
# of qubits.


qubits = [1, 2, 3, 4]
variances_grant = []
variances_random = []
gate_set = [qml.RX, qml.RY, qml.RZ]

for num_qubits in qubits:
    grad_vals_random = []
    grad_vals_grant = []

    for i in range(num_samples):
        # Random initialization
        dev1 = qml.device("default.qubit", wires=num_qubits)
        qcircuit1 = qml.QNode(rand_circuit, dev1)
        grad1 = qml.grad(qcircuit1, argnum=0)

        random_gate_sequence = {i: np.random.choice(gate_set) for i in range(num_qubits)}

        random_initialization = np.random.uniform(0, np.pi, size=(2, num_qubits))
        gradient1 = grad1(
            random_initialization, random_gate_sequence=random_gate_sequence, num_qubits=num_qubits
        )
        grad_vals_random.append(gradient1[0])

        # Grant initialization
        dev2 = qml.device("default.qubit", wires=num_qubits)
        qcircuit2 = qml.QNode(rand_circuit, dev2)
        grad2 = qml.grad(qcircuit2, argnum=0)

        random_params = np.random.uniform(0, np.pi, size=(num_qubits))
        


        grant_initialization = np.array([random_params, -random_params])
        gradient2 = grad2(
            grant_initialization, random_gate_sequence=random_gate_sequence, num_qubits=num_qubits
        )
        grad_vals_grant.append(gradient2[0])

    variances_random.append(np.var(grad_vals_random))
    variances_grant.append(np.var(grad_vals_grant))


variances_random = np.array(variances_random)
variances_grant = np.array(variances_grant)

print("Variances with random init", variances_random)
print("Variances with Grant init", variances_grant)

qubits = np.array(qubits)


# Fit the semilog plot to a straight line
p1 = np.polyfit(qubits, np.log(variances_random), 1)
p2 = np.polyfit(qubits, np.log(variances_grant), 1)

# Plot the straight line fit to the semilog

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].semilogy(qubits, variances_random, "o", label="Random init", color="r")
ax[0].semilogy(qubits, np.exp(p1[0] * qubits + p1[1]), "o-.", color="r")


ax[1].semilogy(qubits, variances_grant, "o", label="Grant init", color="g")
# ax[1].semilogy(qubits, np.exp(p2[0] * qubits + p2[1]), "o-.", color="g")

ax[0].set_xlabel(r"N Qubits")
ax[0].set_ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance")

ax[1].set_xlabel(r"N Qubits")
ax[1].set_ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance")

ax[0].legend()
ax[1].legend()
plt.show()


##############################################################################
# This tutorial was generated using the following PennyLane version:

qml.about()


##############################################################################
# References
# ----------
#
# 1. Dauphin, Yann N., et al.,
#    Identifying and attacking the saddle point problem in high-dimensional non-convex
#    optimization. Advances in Neural Information Processing
#    systems (2014).
#
# 2. McClean, Jarrod R., et al.,
#    Barren plateaus in quantum neural network training landscapes.
#    Nature communications 9.1 (2018): 4812.
#
# 3. Grant, Edward, et al.
#    An initialization strategy for addressing barren plateaus in
#    parametrized quantum circuits. arXiv preprint arXiv:1903.05076 (2019).
