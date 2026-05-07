# adjusted_final_score — sequencing and assembly

Both verifiers emit `adjusted_final_score` alongside the raw `final_score`. The
adjusted form is what's compared across systems.

## sequencing — multiplicative composite

```
adjusted_final_score = nd_score * lis_score * adj_score
```

Any zero component zeros the headline (harsher than the weighted-sum
`final_score`).

## assembly — chance-floor rescale

```
adjusted_final_score = max(0, (final_score - 1/3) * 1.5)
```

Maps random-guess (≈1/3 per-slot accuracy) to 0 and perfect to 1, floored at 0.

| `final_score` | `adjusted_final_score` |
|---:|---:|
| ≤ 0.333 | 0.000 |
| 0.500 | 0.250 |
| 0.667 | 0.500 |
| 0.750 | 0.625 |
| 1.000 | 1.000 |
