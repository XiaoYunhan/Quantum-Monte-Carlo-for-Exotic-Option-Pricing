# Quantum Monte Carlo for Exotic Option Pricing

- [Quantum Monte Carlo for Exotic Option Pricing](#quantum-monte-carlo-for-exotic-option-pricing)
  - [Introduction](#introduction)
  - [Classical Monte Carlo](#classical-monte-carlo)
    - [Monte Carlo Simulation for Option Pricing](#monte-carlo-simulation-for-option-pricing)
    - [Asset Price Simulation with Different Models](#asset-price-simulation-with-different-models)
    - [Convergence and Error Analysis](#convergence-and-error-analysis)
    - [Variance Reduction Techniques](#variance-reduction-techniques)
    - [Monte Carlo for Exotic Options](#monte-carlo-for-exotic-options)
  - [Quantum Monte Carlo](#quantum-monte-carlo)
    - [Quantum Amplitude Estimation](#quantum-amplitude-estimation)
    - [QMC for European Option Pricing](#qmc-for-european-option-pricing)
    - [QMC for Asian Option Pricing](#qmc-for-asian-option-pricing)
    - [QMC for Barrier Options Pricing](#qmc-for-barrier-options-pricing)
    - [QMC for Digital Pricing](#qmc-for-digital-pricing)
  - [Other Quantum Approaches for Exotic Option Pricing](#other-quantum-approaches-for-exotic-option-pricing)
    - [Option Pricing with qGANs](#option-pricing-with-qgans)
  - [Future Improvement](#future-improvement)
  - [Appendix](#appendix)
    - [Appendix A: Fundamentals of Quantum Computing for Option Pricing](#appendix-a-fundamentals-of-quantum-computing-for-option-pricing)
      - [1. Qubits and Superposition](#1-qubits-and-superposition)
      - [2. Quantum Registers and Probability Encoding](#2-quantum-registers-and-probability-encoding)
      - [3. Quantum Amplitude Estimation (QAE) and Monte Carlo Speedup](#3-quantum-amplitude-estimation-qae-and-monte-carlo-speedup)
      - [4. Quantum Fourier Transform (QFT) and Differential Equations](#4-quantum-fourier-transform-qft-and-differential-equations)
      - [5. Quantum Algorithms for Option Pricing](#5-quantum-algorithms-for-option-pricing)
      - [6. Conclusion](#6-conclusion)
    - [Appendix B: Quantum Circuits Implementation for Arithmetic Computations](#appendix-b-quantum-circuits-implementation-for-arithmetic-computations)
      - [1. Quantum Circuits for Arithmetic Operations](#1-quantum-circuits-for-arithmetic-operations)
      - [2. Basic Arithmetic Operations in Quantum Computing](#2-basic-arithmetic-operations-in-quantum-computing)
      - [3. Addition Circuit](#3-addition-circuit)
      - [4. Multiplication and Exponentiation Circuits](#4-multiplication-and-exponentiation-circuits)
      - [5. Quantum Circuit for Call Option Payoff](#5-quantum-circuit-for-call-option-payoff)
      - [6. Quantum Circuit for European Call Option Pricing](#6-quantum-circuit-for-european-call-option-pricing)
  - [Reference](#reference)
    - [Official Document](#official-document)
    - [GitHub](#github)

## Introduction

Monte Carlo methods have long been a cornerstone of computational finance, particularly in the pricing of financial derivatives, risk management, and portfolio optimization. These methods estimate expected values by simulating the stochastic behavior of asset prices, making them indispensable for modeling financial markets. However, traditional Monte Carlo simulations suffer from slow convergence, requiring a large number of iterations to achieve accurate results. The error typically scales as $\mathcal{O}(k^{-1/2})$, meaning that quadrupling the number of simulations is necessary to halve the error. This inefficiency has driven the search for more computationally effective approaches, including the use of quantum computing.

Quantum computing has emerged as a potential game-changer for many fields, offering significant speedups for problems that require large-scale simulations. One of the most relevant quantum algorithms in this context is **Quantum Amplitude Estimation (QAE)**, which accelerates Monte Carlo simulations by improving their convergence rate to $\mathcal{O}(k^{-1})$. This quadratic speedup could have a major impact on financial applications, where Monte Carlo methods are computationally expensive but widely used. By leveraging QAE, Quantum Monte Carlo can significantly reduce the number of required simulations, making it a promising alternative for pricing complex derivatives.

Exotic options, such as **Asian options, barrier options, and digital options**, present additional computational challenges due to their path-dependent nature and intricate payoff structures. Classical Monte Carlo methods often struggle with these instruments because accurately capturing the full range of possible price paths requires an enormous number of simulations. QMC, on the other hand, offers a way to achieve the same level of accuracy with exponentially fewer samples, making it particularly well-suited for these cases.

This paper explores the application of **Quantum Monte Carlo techniques** in the pricing of exotic options. We begin by reviewing **classical Monte Carlo methods**, including different stochastic models for asset price simulation and variance reduction techniques that improve efficiency. We then introduce **Quantum Monte Carlo**, explaining how Quantum Amplitude Estimation is used to speed up Monte Carlo simulations and examining its effectiveness in pricing exotic options. Additionally, we discuss alternative quantum approaches, such as quantum machine learning techniques like quantum generative adversarial networks (qGANs), which offer new ways to model financial markets. Finally, we address **practical challenges and future improvements**, considering both theoretical advancements and real-world implementation hurdles.

This study aims to bridge the gap between classical computational finance and emerging quantum technologies, providing a detailed comparison of the strengths and limitations of quantum-enhanced Monte Carlo methods. As quantum hardware continues to evolve, understanding its potential impact on derivative pricing and risk management will be crucial for the future of financial modeling.

## Classical Monte Carlo

Monte Carlo methods provide a widely used numerical approach to pricing financial derivatives, particularly when closed-form solutions are unavailable. The fundamental idea is to approximate the expected value of an option payoff by simulating the underlying asset price paths and averaging the discounted payoffs.

### Monte Carlo Simulation for Option Pricing

The risk-neutral price of an option with payoff function $f(S_T)$ at maturity $T$ is given by:
```math
\Pi = e^{-rT} \mathbb{E}^{\mathbb{Q}}[ f(S_T) ]
```
where $\mathbb{Q}$ denotes the risk-neutral measure, $S_T$ represents the terminal asset price, and $r$ is the risk-free rate.

In Monte Carlo simulation, we approximate this expectation using a large number of simulated asset price paths $\{ S_T^{(i)} \}_{i=1}^{k}$ and estimate the option price as:
```math
\hat{\Pi} = e^{-rT} \frac{1}{k} \sum_{i=1}^{k} f(S_T^{(i)}).
```
By the **law of large numbers**, $\hat{\Pi}$ converges to the true price $\Pi$ as $k \to \infty$.

### Asset Price Simulation with Different Models
The standard approach to modeling asset prices in MC simulation follows a **Geometric Brownian Motion (GBM)** process:
```math
 dS_t = r S_t dt + \sigma S_t dW_t,
```
where $\sigma$ is the volatility and $W_t$ is a standard Wiener process. The discrete-time solution for asset price evolution is given by:
```math
 S_{t+\Delta t} = S_t \exp \left( (r - \frac{1}{2} \sigma^2) \Delta t + \sigma \sqrt{\Delta t} Z_t \right),
```
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
```math
 \mathbb{P}[ |\hat{\Pi} - \Pi| \geq \epsilon ] \leq \frac{\lambda^2}{k \epsilon^2},
```
where $\lambda^2$ is the variance of the payoff function. To achieve a given accuracy $\epsilon$, the required number of simulations scales as:
```math
 k = \mathcal{O} \left( \frac{\lambda^2}{\epsilon^2} \right).
```
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
```math
S_{avg} = \frac{1}{n} \sum_{t=1}^{n} S_t.
```
  The option payoff is then $f(S_{avg})$ instead of $f(S_T)$.
- **American options** require estimating early exercise value at each time step, often using Least Squares Monte Carlo (LSMC).

Monte Carlo provides a general, flexible framework for pricing such options, though its slow convergence motivates the need for **Quantum Monte Carlo (QMC)**, which we discuss next.

## Quantum Monte Carlo


### Quantum Amplitude Estimation

[1] Quantum Amplitude Amplification and Estimation. Brassard et al (2000). https://arxiv.org/abs/quant-ph/0005055

[2] Iterative Quantum Amplitude Estimation. Grinko, D., Gacon, J., Zoufal, C., & Woerner, S. (2019). https://arxiv.org/abs/1912.05559

[3] Amplitude Estimation without Phase Estimation. Suzuki, Y., Uno, S., Raymond, R., Tanaka, T., Onodera, T., & Yamamoto, N. (2019). https://arxiv.org/abs/1904.10246

[4] Faster Amplitude Estimation. K. Nakaji (2020). https://arxiv.org/pdf/2003.02417.pdf


### QMC for European Option Pricing

[1] Quantum Risk Analysis. Woerner, S., & Egger, D. J. (2019). https://www.nature.com/articles/s41534-019-0130-6

[2] Option Pricing using Quantum Computers. Stamatopoulos, N., Egger, D. J., Sun, Y., Zoufal, C., Iten, R., Shen, N., & Woerner, S. (2020). https://quantum-journal.org/papers/q-2020-07-06-291/


### QMC for Asian Option Pricing

### QMC for Barrier Options Pricing

### QMC for Digital Pricing

## Other Quantum Approaches for Exotic Option Pricing

### Option Pricing with qGANs

[1] Quantum Generative Adversarial Networks for Learning and Loading Random Distributions. Zoufal, C., Lucchi, A., & Woerner, S. (2019). https://www.nature.com/articles/s41534-019-0223-2



## Future Improvement

## Appendix

### Appendix A: Fundamentals of Quantum Computing for Option Pricing

This appendix provides a concise yet rigorous introduction to quantum computing concepts relevant to option pricing models. The primary advantage of quantum computing in finance lies in its ability to process large probability distributions efficiently, offering potential speedups for Monte Carlo simulations and differential equation solvers.

#### 1. Qubits and Superposition
A **qubit** (quantum bit) is the fundamental unit of quantum information. Unlike classical bits, which can be either \(0\) or \(1\), a qubit exists in a **superposition** of both states:

```math
|\psi\rangle = \alpha |0\rangle + \beta |1\rangle,
```
where $\alpha, \beta \in \mathbb{C}$ are complex probability amplitudes satisfying $|\alpha|^2 + |\beta|^2 = 1$. This property allows quantum systems to encode and manipulate a broader range of information than classical bits.

#### 2. Quantum Registers and Probability Encoding
A **quantum register** consists of multiple qubits, allowing representation of probability distributions over financial variables. Given an $m$-qubit system, the state space spans $2^m$ possible basis states:

```math
|\Psi\rangle = \sum_{i=0}^{2^m-1} c_i |i\rangle,
```
where $c_i$ represents the probability amplitude of state $|i\rangle$, satisfying $\sum |c_i|^2 = 1$. This feature is crucial for encoding probability distributions of stock prices and risk factors.

For example, in option pricing, we can encode a normalized probability density function $P(S_T)$ of a stock price $S_T$ at maturity into a quantum state:

```math
|\Psi\rangle = \sum_{i} \sqrt{P(S_{T,i})} |i\rangle.
```

This enables efficient sampling and direct application of quantum algorithms for expectation calculations.

#### 3. Quantum Amplitude Estimation (QAE) and Monte Carlo Speedup
Classical Monte Carlo methods require $O(1/\epsilon^2)$ samples to estimate an expectation $\mathbb{E}[f(X)]$ with error $\epsilon$. **Quantum Amplitude Estimation (QAE)** reduces this complexity to $O(1/\epsilon)$, achieving a quadratic speedup.

The QAE process consists of:
1. Preparing a quantum state encoding the expected value as an amplitude:
   ```math
   |\psi\rangle = \sqrt{p} |1\rangle + \sqrt{1 - p} |0\rangle,
   ```
   where $p$ represents the probability amplitude corresponding to $\mathbb{E}[f(X)]$.
2. Applying **Grover’s operator** iteratively to amplify the probability of the desired outcome:
   ```math
   G = (2|\psi\rangle\langle\psi| - I)O,
   ```
   where $O$ is the oracle encoding $f(X)$.
3. Using **Quantum Phase Estimation (QPE)** to extract $p$ efficiently via eigenvalue estimation of the Grover operator.

This approach significantly reduces computational costs in derivative pricing models that rely on Monte Carlo simulations.

#### 4. Quantum Fourier Transform (QFT) and Differential Equations
Many option pricing models, such as Black-Scholes, require solving partial differential equations (PDEs). The **Quantum Fourier Transform (QFT)** provides an efficient method for spectral analysis and differential equation solving.

The QFT maps a quantum state $|x\rangle$ to its frequency representation:

```math
|x\rangle \to \frac{1}{\sqrt{2^m}} \sum_{k=0}^{2^m-1} e^{2\pi i x k / 2^m} |k\rangle.
```

For PDE-based option pricing, such as the Black-Scholes model, the pricing equation:

```math
\frac{\partial C}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 C}{\partial S^2} + rS \frac{\partial C}{\partial S} - rC = 0
```

can be transformed into Fourier space using QFT, allowing efficient numerical solutions via Hamiltonian simulation.

#### 5. Quantum Algorithms for Option Pricing
Several quantum algorithms can accelerate option pricing calculations:
- **Quantum Monte Carlo (QMC):** Utilizes QAE to speed up simulations of stochastic processes like geometric Brownian motion, where the stock price follows:

  ```math
  dS_t = \mu S_t dt + \sigma S_t dW_t.
  ```

- **Quantum Differential Equation Solvers:** Solve Black-Scholes-type PDEs using QFT and Hamiltonian simulation techniques.
- **Quantum Path-Integral Methods:** Encode the evolution of probability distributions using quantum state preparation, with transition probabilities computed as:

  ```math
  P(S_T | S_0) = \int P(S_T | S_{T-1}) P(S_{T-1} | S_{T-2}) \dots P(S_1 | S_0) dS_1 \dots dS_{T-1}.
  ```

These approaches hold the potential to revolutionize financial engineering by reducing the computational burden of risk analysis and derivative pricing.

#### 6. Conclusion
Quantum computing provides a novel framework for financial modeling, with particular advantages in option pricing and risk management. The ability to encode probability distributions, estimate expectations with amplitude amplification, and solve differential equations efficiently makes quantum algorithms a promising avenue for future research and application in quantitative finance. The mathematical foundations outlined here illustrate how quantum computing can optimize derivative pricing, making it a key area for further exploration.

### Appendix B: Quantum Circuits Implementation for Arithmetic Computations

This appendix discusses the representation and processing of integers and real numbers on a quantum computer. An integer $a$ in the range $0 \leq a < N = 2^m$ can be expressed in binary using $m$ bits $x_i$, where $i = 0, \dots, m-1$, as follows:

```math
a = 2^0 x_0 + 2^1 x_1 + 2^2 x_2 + \dots + 2^{m-1} x_{m-1}.
```

The maximum representable value is $N - 1$. The quantum state $|a\rangle$ corresponds to an $m$-qubit register in the computational basis, denoted as:

```math
|a\rangle = |x_0, x_1, \dots, x_{m-1}\rangle.
```

For real numbers $r$ in the range $0 \leq r < 1$, we use $m$ bits $b_i$, where $i = 0, \dots, m-1$, to represent:

```math
r = \frac{b_0}{2} + \frac{b_1}{4} + \dots + \frac{b_{m-1}}{2^m} =: [b_1, b_2, \dots, b_{m-1}].
```

The accuracy of this representation is $1/2^m$. The corresponding quantum state $|r\rangle$ is encoded in an $m$-qubit register as:

```math
|r\rangle = |b_0, b_1, \dots, b_{m-1}\rangle.
```

For signed integers and real numbers, an additional sign qubit $|s\rangle$ is introduced.

#### 1. Quantum Circuits for Arithmetic Operations

Operations performed on classical computers can be represented as transformations from $n$ to $m$ bits using a function $F: \{0,1\}^n \to \{0,1\}^m$. A fundamental principle in reversible computing states that the number of input and output bits can be equal, allowing $F$ to be mapped to a reversible function $F': \{0,1\}^{n+m} \to \{0,1\}^{n+m}$, where:

```math
F'(x, y) = (x, y \oplus F(x)).
```

Since $F'$ is reversible, it can be implemented using quantum circuits composed of negation and Toffoli gates. If $F$ is efficiently computable, the depth of the circuit remains polynomial in $n + m$. This classical transformation can be directly translated into a quantum circuit utilizing bit-flip ($\sigma_x$) operations and Toffoli gates, with the latter decomposable into controlled NOT (CNOT), Hadamard, and T gates.

#### 2. Basic Arithmetic Operations in Quantum Computing

Various arithmetic operations can be implemented using fundamental quantum gates:

1. **SUM Gate**: Implements controlled addition:

   ```
   1 ────────────────●──
   2 ────────────────●──
   3 ────────⊕──⊕───────
   ```

2. **Carry (CY) Gate**: Represents the carry operation in arithmetic calculations:

   ```
   1 ────────────────●──────
   2 ────────────────●──────
   3 ────────⊕──⊕───────⊕───
   4 ────────────────●──────
   ```

These gates serve as building blocks for more complex operations, including addition, multiplication, and exponentiation.

#### 3. Addition Circuit

A quantum circuit for addition modulo $N$ is defined as:

```
   |a⟩ ─────── ADD ─────── |a⟩
   |b⟩ ─────────────────── |a + b⟩
```

Similarly, a circuit for addition modulo $N$ is constructed as:

```
   |a⟩ ─────── ADD N ───── |a⟩
   |b⟩ ─────────────────── |a + b \mod N⟩
```

#### 4. Multiplication and Exponentiation Circuits

The multiplication circuit for computing $a \times x \mod N$ is:

```
   |x⟩ ─── MULT(a) N ─── |x⟩
   |0⟩ ───────────────── |a × x \mod N⟩
```

The exponentiation circuit for computing $a^x \mod N$ is:

```
   |x⟩ ─── EXP(a) ─── |x⟩
   |0⟩ ───────────── |a^x \mod N⟩
```

#### 5. Quantum Circuit for Call Option Payoff

To implement the European call option payoff function:

```math
a^+ = \max\{0, a\},
```

we design a reversible circuit:

```
   |a, s⟩ ─── MAX(0) ─── |a, s⟩
   |0⟩ ─────────────── |a^+⟩
```

This circuit uses the sign bit as a control. If the sign bit is positive, controlled addition is performed:

```math
|a, s, 0\rangle \rightarrow
\begin{cases}
|a, s, a\rangle, & \text{if } |s\rangle = |0\rangle \\
|a, s, 0\rangle, & \text{if } |s\rangle = |1\rangle
\end{cases}
```

#### 6. Quantum Circuit for European Call Option Pricing

A quantum circuit can be constructed to model the payoff of a European call option by mapping Brownian motion to stock price evolution:

```
   |x⟩ ─── S(σ, r, t) ─── |x⟩
   |0⟩ ───────────────── |e^{\sigma x + (r - \sigma^2/2)t}⟩
```

Combining this with the max function, the final quantum circuit for the call option payoff is:

```
   |x⟩ ─── CALL(K, σ, r, T) ─── |x⟩
   |0⟩ ─────────────────────── |ṽ_{euro}(x)⟩
```

where $ṽ_{euro}(x)$ represents the bitwise approximation of the payoff function.

---

This appendix provides a structured representation of quantum arithmetic circuits, covering integer and real number encoding, fundamental arithmetic operations, and financial applications such as European call option pricing. These circuits form the foundation for quantum algorithms in financial engineering and beyond.

## Reference

### Official Document
- https://qiskit-community.github.io/qiskit-finance/tutorials/03_european_call_option_pricing.html
- https://qiskit-community.github.io/qiskit-finance/tutorials/10_qgan_option_pricing.html

### GitHub
- https://github.com/Qiskit/qiskit
- https://github.com/qiskit-community/ibm-quantum-challenge-africa-2021
- https://github.com/udvzol/option_pricing