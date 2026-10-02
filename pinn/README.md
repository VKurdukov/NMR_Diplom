# Physics-informed NMR inversion

This package compares a positive neural inverse model with the non-negative
Tikhonov method used in the diploma. The neural network represents
`f(B_loc)`; the known NMR integral operator is embedded in its loss.

## Windows launch without uv commands

One-time setup: double-click [setup_pinn.bat](../setup_pinn.bat). It creates
the isolated `.venv` directory and installs the packages. `uv` is used only in
this setup step; it is not needed for regular work.

Then double-click one of these files in the project root:

- `run_synthetic_tests.bat` — synthetic benchmark;
- `run_real_data_test.bat` — all spectra from `test_data`.

They invoke `.venv\Scripts\python.exe`, so they are ordinary Python launches
with fixed project-local dependencies. The command window remains open to show
progress and errors.

The synthetic benchmark includes Gaussian, Lorentzian, their mixture, two
Gaussians, an edge peak, a step distribution and a grid-resolved delta peak.
It adds reproducible Gaussian noise and compares PINN with the diploma's exact
Tikhonov implementation.
