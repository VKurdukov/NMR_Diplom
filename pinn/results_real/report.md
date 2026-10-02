# FieldSweep: PINN vs diploma Tikhonov

Both methods were fitted to each normalized experimental spectrum independently.
The Tikhonov baseline exactly follows `reconstruct_field_distribution.py`: 70 nodes, unweighted kernel, its D matrix, non-negativity, free background and its GCV rule.
PINN uses the same no-convolution kernel, but treats its output as a normalized density and includes quadrature in the integral.

PINN has lower in-sample spectral RMSE in 22 of 22 temperatures.

| T, K | N | Tikhonov RMSE | PINN RMSE | lambda |
|---:|---:|---:|---:|---:|
| 3.50 | 71 | 0.03424 | 0.03342 | 5.722e+05 |
| 4.50 | 71 | 0.03653 | 0.03553 | 4.535e+05 |
| 5.50 | 71 | 0.04081 | 0.03953 | 5.722e+05 |
| 6.20 | 71 | 0.04176 | 0.04054 | 5.722e+05 |
| 6.90 | 71 | 0.03927 | 0.03800 | 5.722e+05 |
| 7.50 | 71 | 0.03487 | 0.03387 | 4.535e+05 |
| 8.00 | 71 | 0.03542 | 0.03440 | 4.535e+05 |
| 8.50 | 71 | 0.03229 | 0.03156 | 7.221e+05 |
| 9.00 | 71 | 0.02788 | 0.02670 | 5.722e+05 |
| 9.50 | 71 | 0.02523 | 0.02427 | 1.789e+05 |
| 10.00 | 71 | 0.02540 | 0.02455 | 1.417e+05 |
| 10.50 | 58 | 0.02990 | 0.02882 | 1.789e+05 |
| 11.00 | 35 | 0.01590 | 0.01470 | 4.431e+04 |
| 11.50 | 35 | 0.01685 | 0.01567 | 4.431e+04 |
| 12.00 | 35 | 0.01751 | 0.01649 | 3.511e+04 |
| 12.50 | 27 | 0.01025 | 0.00940 | 1.385e+04 |
| 13.00 | 27 | 0.01060 | 0.00973 | 1.385e+04 |
| 14.00 | 27 | 0.01076 | 0.00977 | 1.385e+04 |
| 15.00 | 27 | 0.01086 | 0.01011 | 1.097e+04 |
| 17.00 | 27 | 0.01073 | 0.00982 | 1.097e+04 |
| 20.00 | 27 | 0.01012 | 0.00932 | 8.697e+03 |
| 25.00 | 27 | 0.00835 | 0.00758 | 6.893e+03 |