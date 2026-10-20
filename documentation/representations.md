# Hidden representations of the RBM

`QELM.NQS.evaluation.representations` looks at what the hidden layer of an RBM has learned:

```text
ψ(s) = exp(Σ_i a_i s_i) Π_j 2 cosh(b_j + Σ_i W_ij s_i)
```

`W` couples spin `i` to hidden unit `j`. NetKet stores it as `parameters["Dense"]["kernel"]` with shape `(N, M)`, `M = α N` (visible by hidden).

```python
from QELM.NQS.evaluation.representations import analyze_weights_svd, extract_rbm_weights
from QELM.NQS.utils.makeplots import plot_singular_values

W = extract_rbm_weights(state)  # (N, M), complex for a complex RBM
svd = analyze_weights_svd(W)  # W = U diag(σ) Vh, thin SVD
svd.singular_values  # real, ≥ 0, descending
svd.U[:, 0]  # dominant spin pattern
svd.weight_fractions  # σ_k² / Σ σ², share of ||W||_F²
svd.n_dominant_modes(0.9)  # modes carrying 90 % of ||W||_F²
plot_singular_values(svd.singular_values)  # written to the run's plots/
```

Relabelling the hidden units leaves `ψ` unchanged (a gauge symmetry of the RBM), and it leaves the spectrum unchanged too. A test checks both. So `σ` describes the state, not the arbitrary order of the hidden units. The columns of `U` are spin patterns. A rank-one `W` built on a staggered pattern, for instance, gives one mode whose `U[:, 0]` is that Néel pattern.
