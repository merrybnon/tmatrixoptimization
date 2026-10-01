# 1000andUniform

Tom1000 with the uniform matrix added: U_ij = 1/19 off the diagonal, 0 on it, target
2400. Derived, not received: `src/tm_ml/uniform/make_drop.py` writes it from
`data/raw/Tom1000/Analysis_Results.npz`, and rerunning it reproduces this file.

U is at index 1, the first position in the training split for seeds 0, 1 and 2 at
val and test fractions of 0.1. The other 1000 matrices and targets are Tom1000's,
in Tom1000's order. Adding a matrix changes n, so every split differs from Tom1000's
at the same seed.
