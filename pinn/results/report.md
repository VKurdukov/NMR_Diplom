# PINN and Tikhonov synthetic benchmark

Noise standard deviation: `0.0050` of normalized spectrum amplitude.

| Case | Method | Distribution rel. L2 | Spectrum RMSE (clean) | W1, T |
|---|---|---:|---:|---:|
| gaussian | Tikhonov (diploma) | 0.0195 | 0.00149 | 0.000241 |
| gaussian | PINN | 0.0780 | 0.00200 | 0.000282 |
| lorentzian | Tikhonov (diploma) | 0.0711 | 0.00289 | 0.000649 |
| lorentzian | PINN | 0.1216 | 0.00342 | 0.000712 |
| mixture | Tikhonov (diploma) | 0.0689 | 0.00230 | 0.000573 |
| mixture | PINN | 0.1100 | 0.00263 | 0.000746 |
| double_gaussian | Tikhonov (diploma) | 0.0606 | 0.00234 | 0.000295 |
| double_gaussian | PINN | 0.1237 | 0.00196 | 0.000357 |
| edge_peak | Tikhonov (diploma) | 0.0465 | 0.00158 | 0.000240 |
| edge_peak | PINN | 0.0483 | 0.00149 | 0.000156 |
| step | Tikhonov (diploma) | 0.1676 | 0.00372 | 0.000413 |
| step | PINN | 0.1139 | 0.00230 | 0.000292 |
| delta | Tikhonov (diploma) | 0.4932 | 0.00190 | 0.001381 |
| delta | PINN | 0.4863 | 0.00180 | 0.000817 |

The Tikhonov baseline is an exact reproduction of the diploma's discrete second-order penalty and GCV rule. PINN uses the normalized integral operator.