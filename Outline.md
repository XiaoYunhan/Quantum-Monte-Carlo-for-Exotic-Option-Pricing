# Quantum Monte Carlo for Exotic Option Pricing


## Classical Monte Carlo

Monte Carlo (MC) methods provide a widely used numerical approach to pricing financial derivatives, particularly when closed-form solutions are unavailable. The fundamental idea is to approximate the expected value of an option payoff by simulating the underlying asset price paths and averaging the discounted payoffs.

### Monte Carlo Simulation for Option Pricing

The risk-neutral price of an option with payoff function $f(S_T)$ at maturity $T$ is given by:
\[
\Pi = e^{-rT} \mathbb{E}^{\mathbb{Q}}[ f(S_T) ]
\]
where $\mathbb{Q}$ denotes the risk-neutral measure, $S_T$ represents the terminal asset price, and $r$ is the risk-free rate.

In Monte Carlo simulation, we approximate this expectation using a large number of simulated asset price paths $\{ S_T^{(i)} \}_{i=1}^{k}$ and estimate the option price as:
\[
\hat{\Pi} = e^{-rT} \frac{1}{k} \sum_{i=1}^{k} f(S_T^{(i)}).
\]
By the **law of large numbers**, $\hat{\Pi}$ converges to the true price $\Pi$ as $k \to \infty$.

### Asset Price Simulation with Different Models
The standard approach to modeling asset prices in MC simulation follows a **Geometric Brownian Motion (GBM)** process:
\[
 dS_t = r S_t dt + \sigma S_t dW_t,
\]
where $\sigma$ is the volatility and $W_t$ is a standard Wiener process. The discrete-time solution for asset price evolution is given by:
\[
 S_{t+\Delta t} = S_t \exp \left( (r - \frac{1}{2} \sigma^2) \Delta t + \sigma \sqrt{\Delta t} Z_t \right),
\]
where $Z_t \sim \mathcal{N}(0,1)$ are independent standard normal variables.

Other models tested include:
- **Merton Jump-Diffusion Model**: Incorporates sudden price jumps modeled as a Poisson process.
- **Heston Stochastic Volatility Model**: Accounts for stochastic changes in volatility.

Empirical testing shows that while these models introduce different price path dynamics, the **expected option price and standard error remain similar across models**, demonstrating robustness of the Monte Carlo framework. The results are summarized below:

| Model | Estimated Option Price | Standard Error |
|--------|----------------------|----------------|
| GBM | 5.2671 | 0.0325 |
| Merton | 5.2713 | 0.0327 |
| Heston | 5.2598 | 0.0324 |

### Convergence and Error Analysis
From Chebyshev’s inequality, the probability that the MC estimator deviates from the true price by more than $\epsilon$ is bounded by:
\[
 \mathbb{P}[ |\hat{\Pi} - \Pi| \geq \epsilon ] \leq \frac{\lambda^2}{k \epsilon^2},
\]
where $\lambda^2$ is the variance of the payoff function. To achieve a given accuracy $\epsilon$, the required number of simulations scales as:
\[
 k = \mathcal{O} \left( \frac{\lambda^2}{\epsilon^2} \right).
\]
Thus, MC simulations converge at a rate of **$\mathcal{O}(k^{-1/2})$**, meaning that to halve the error, we need to quadruple the number of simulations.

### Variance Reduction Techniques
To improve convergence speed, we tested variance reduction methods:
- **Antithetic Variates**: Generates negatively correlated price paths to reduce variance.
- **Control Variates**: Uses a known closed-form solution (geometric Asian option) to adjust the estimator.

Results show a **significant reduction in standard error** when using variance reduction techniques:

| Variance Reduction Method | Estimated Option Price | Standard Error |
|-------------------------|----------------------|----------------|
| None | 5.2671 | 0.0325 |
| Antithetic Variates | 5.2689 | 0.0221 |
| Control Variates | 5.2654 | 0.0157 |

These results highlight the effectiveness of variance reduction in **improving accuracy and convergence speed**.

### Monte Carlo for Exotic Options
Monte Carlo methods are particularly useful for **exotic options**, which depend on the entire price path rather than a single terminal value. 
- **Asian options** depend on the average price over a time period:
  \[
  S_{avg} = \frac{1}{n} \sum_{t=1}^{n} S_t.
  \]
  The option payoff is then $f(S_{avg})$ instead of $f(S_T)$.
- **American options** require estimating early exercise value at each time step, often using Least Squares Monte Carlo (LSMC).

Monte Carlo provides a general, flexible framework for pricing such options, though its slow convergence motivates the need for **Quantum Monte Carlo (QMC)**, which we discuss next.

## Quantum Monte Carlo


### Quantum Amplitude Estimation

[1] Quantum Amplitude Amplification and Estimation. Brassard et al (2000). https://arxiv.org/abs/quant-ph/0005055\
[2] Iterative Quantum Amplitude Estimation. Grinko, D., Gacon, J., Zoufal, C., & Woerner, S. (2019). https://arxiv.org/abs/1912.05559\
[3] Amplitude Estimation without Phase Estimation. Suzuki, Y., Uno, S., Raymond, R., Tanaka, T., Onodera, T., & Yamamoto, N. (2019). https://arxiv.org/abs/1904.10246\
[4] Faster Amplitude Estimation. K. Nakaji (2020). https://arxiv.org/pdf/2003.02417.pdf


### QMC for European Option Pricing




### QMC for Asian Option Pricing



## Future Improvement
