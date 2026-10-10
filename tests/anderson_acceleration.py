"""Check Anderson acceleration on fixed-point problems with known solutions.

Usage: python anderson_acceleration.py
"""

import numpy as np

from fabmos.anderson import AndersonAccelerator

rng = np.random.default_rng(1)
n = 400


def solve(g, x, accelerator, tolerance=1e-10, max_evaluations=5000):
    """Iterate until the residual g(x) - x is small. Returns x and the number of
    evaluations of g."""
    for evaluations in range(1, max_evaluations + 1):
        gx = g(x)
        if np.linalg.norm(gx - x) <= tolerance * np.linalg.norm(gx):
            break
        x = accelerator.update(x, gx)
    return x, evaluations


def error(x, reference):
    return np.linalg.norm(x - reference) / np.linalg.norm(reference)


# A linear map with three slowly decaying modes, like a model with a deep ocean
Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
eigenvalues = np.concatenate(([0.999, 0.995, 0.99], rng.uniform(0.0, 0.9, n - 3)))
A = (Q * eigenvalues) @ Q.T
b = rng.standard_normal(n)
fixed_point = np.linalg.solve(np.eye(n) - A, b)


def linear(x):
    return A @ x + b


x, evaluations = solve(linear, np.zeros(n), AndersonAccelerator(m=50))
assert evaluations < 100, f"Not converged after {evaluations} evaluations"
assert error(x, fixed_point) < 1e-8, f"Wrong solution: error {error(x, fixed_point)}"
plain = np.zeros(n)
for _ in range(evaluations):
    plain = linear(plain)
assert error(plain, fixed_point) > 0.1, "The problem is too easy to test anything"
print(
    f"OK: linear problem solved in {evaluations} evaluations (error"
    f" {error(x, fixed_point):.1e}; plain iteration: {error(plain, fixed_point):.1e})"
)

# Without memory, the result is the plain iteration, to the last bit
accelerator = AndersonAccelerator(m=0)
x = np.zeros(n)
plain = np.zeros(n)
for _ in range(5):
    x = accelerator.update(x, linear(x))
    plain = linear(plain)
assert np.array_equal(x, plain), "m=0 differs from the plain iteration"
print("OK: m=0 reproduces the plain iteration exactly")

# A quantity that the map conserves (here: the sum of all elements, as the columns
# of P sum to one) is conserved by the accelerated iteration too
P = rng.uniform(size=(n, n))
P = 0.02 * P / P.sum(axis=0) + 0.98 * np.eye(n)
x = rng.uniform(size=n)
total = x.sum()
accelerator = AndersonAccelerator(m=10)
largest_change = 0.0
for _ in range(40):
    x = accelerator.update(x, P @ x)
    largest_change = max(largest_change, abs(x.sum() / total - 1.0))
assert largest_change < 1e-10, f"The total changed by {largest_change}"
assert np.linalg.norm(P @ x - x) < 1e-10, "Not converged"
print(f"OK: conserved total changed by at most {largest_change:.1e}")

# Starting from the solution: residuals of exactly zero must not give NaN
accelerator = AndersonAccelerator(m=5)
x = fixed_point.copy()
for _ in range(4):
    x = accelerator.update(x, x.copy())
assert np.array_equal(x, fixed_point), "The solution was not left alone"
print("OK: a converged state stays as it is")

# A nonlinear map
def nonlinear(x):
    return np.cos(0.5 * (A @ x))


x, evaluations = solve(nonlinear, np.zeros(n), AndersonAccelerator(m=10))
plain, plain_evaluations = solve(nonlinear, np.zeros(n), AndersonAccelerator(m=0))
assert error(x, plain) < 1e-7, "Accelerated and plain iteration disagree"
assert evaluations <= plain_evaluations, "Acceleration made it slower"
print(f"OK: nonlinear problem: {evaluations} evaluations, plain {plain_evaluations}")

# Damping still converges to the same solution
x, evaluations = solve(linear, np.zeros(n), AndersonAccelerator(m=20, beta=0.5))
assert error(x, fixed_point) < 1e-6, "Wrong solution with damping"
print(f"OK: with beta=0.5, solved in {evaluations} evaluations")

for kwargs in (dict(m=-1), dict(beta=0.0), dict(beta=1.5)):
    try:
        AndersonAccelerator(**kwargs)
    except ValueError:
        continue
    raise AssertionError(f"{kwargs} was accepted")
print("OK: invalid settings are rejected")
