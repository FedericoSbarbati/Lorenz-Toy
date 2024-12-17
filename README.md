## Mathematical Framework

### Stochastic Lorenz System with Noise
We simulate the **stochastic Lorenz system** with added intrinsic noise, modeled as Wiener processes. The governing equations are:

\[
\begin{aligned}
dz_1 &= 10(z_2 - z_1)dt + \alpha dW_1, \\
dz_2 &= \left[z_1(28 - z_3) - z_2\right]dt + \alpha dW_2, \\
dz_3 &= \left[z_1 z_2 - \frac{8}{3}z_3\right]dt + \alpha dW_3,
\end{aligned}
\]

where \( z_1, z_2, z_3 \) represent the state variables of the Lorenz system, \( \alpha \) scales the dynamical noise, and \( W_1, W_2, W_3 \) are independent Wiener processes.

---

### Observables with Measurement Noise
We construct two observables:

1. \( y_1 \) is a noisy scalar measurement derived from \( z_2 \):
\[
y_1 = z_2 + \beta \sigma_1,
\]
where \( \sigma_1 \sim \mathcal{N}(0, 1) \) is Gaussian noise scaled by \( \beta \).

2. \( y_2 \) is a multidimensional observable derived f
