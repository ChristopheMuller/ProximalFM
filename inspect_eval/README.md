# inspect_eval

`inspect_eval` builds tiny-concept, full-format evaluation sets. It currently
contains one mechanism, `linear_gaussian`, in which `X`, `U`, `Z`, `W`, and the
outcome are scalar and all continuous equations use Normal noise.

The graph is:

```text
X -> U
X -> Z, W, A, Y
U -> Z, W, A, Y
A -> Y
```

There is deliberately no direct `Z -> A` edge and no direct `W -> Y` edge.
All structural coefficients, intercepts, and noise scales are redrawn for each
dataset. Treatment is binary with a linear logistic score, and
`tau(X) = tau_0 + tau_X X`.

Build a point-evaluation set:

```bash
uv run python -m inspect_eval.build_proximal_datasets \
  --name linear_point_v1 --n_max 512 --n_q 128 --seed 123
```

Build a replicate-bearing set for point and distributional evaluation:

```bash
uv run python -m inspect_eval.build_proximal_datasets \
  --name linear_dist_v1 --n_max 512 --n_q 128 --mc_cate 1000 --seed 123
```

Outputs use the same layout as `synt_eval`: the benchmark reads `<root>/val`,
while the trainer receives the build root through `--synt_eval_dir` when
`mc_cate > 1`.
