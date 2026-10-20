# Quantum geometric tensor

`QRC.NQS.evaluation.QGT` analyzes the geometry of the variational manifold. For log-derivatives `O_k(s) = ∂ log ψ_θ(s) / ∂θ_k`, the quantum geometric tensor is

```text
S_kl = <O_k* O_l> − <O_k*><O_l>        (averages over |ψ(s)|²)
```

It is Hermitian and positive semi-definite, and it splits into:

- the **Fubini-Study metric** `g = Re S`, the quantum Fisher information up to a factor 4. It is real symmetric and positive semi-definite, and it measures the distance between nearby states;
- the **Berry curvature** `Ω = 2 Im S`. It is real antisymmetric and gives the geometric phase picked up around a closed loop in parameter space. It is zero for real parameters.

```python
from QRC.NQS.evaluation.QGT import analyze_qgt, qgt_of_state

S = qgt_of_state(state, holomorphic=True)  # dense, no diagonal shift
result = analyze_qgt(S)
result.metric, result.curvature, result.eigenvalues
result.rank  # eigenvalues of g above rtol * max
result.condition_number  # inf when g is singular
result.effective_condition_number  # restricted to the kept directions
result.trace  # total Fisher information
result.curvature_norm  # Frobenius norm of Ω
```

`analyze_qgt` accepts a dense NumPy or JAX array, a `scipy.sparse` matrix, or a NetKet QGT operator (`QGTJacobianDense`, `QGTOnTheFly`, anything with `to_dense()`). Every input gives the same result. A non-Hermitian input is rejected.

## What the RBM looks like at initialization

At `W = b = 0`, `ψ(s) ∝ exp(Σ a_i s_i)` and `O_{b_j} = O_{W_ij} = tanh(0) = 0`. Only the `N` visible biases move the state, so `rank(g) = N` out of `N + M + NM` parameters, and `g_{a_i a_i} = 1 − tanh²(2 Re a_i)`. A test checks this. NetKet initializes the RBM with standard deviation 0.01, which puts it right next to this singular point. This is why the first TDVP step from the default initialization leaves the exact trajectory (see the Loschmidt echo test of issue #8). A small effective rank or a huge condition number signals over-parameterization or a near-singular starting point.
