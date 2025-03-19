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
    - [Quantum Monte Carlo for European Option Pricing](#quantum-monte-carlo-for-european-option-pricing)
      - [Uncertainty Model](#uncertainty-model)
      - [Payoff Function Implementation](#payoff-function-implementation)
      - [Delta Evaluation](#delta-evaluation)
      - [Example Quantum Implementation](#example-quantum-implementation)
    - [Quantum Monte Carlo for Asian Option Pricing](#quantum-monte-carlo-for-asian-option-pricing)
    - [Quantum Monte Carlo for Barrier Options Pricing](#quantum-monte-carlo-for-barrier-options-pricing)
      - [Uncertainty Model](#uncertainty-model-1)
      - [Payoff Function Implementation](#payoff-function-implementation-1)
      - [Quantum Circuit Example](#quantum-circuit-example)
    - [Quantum Monte Carlo for Digital Pricing](#quantum-monte-carlo-for-digital-pricing)
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

### Quantum Monte Carlo for European Option Pricing

Quantum Monte Carlo (QMC) techniques offer a powerful alternative to classical Monte Carlo methods for pricing European call options. Consider a European call option characterized by a strike price $K$ and an underlying asset whose maturity spot price $S_T$ follows a given probability distribution. The payoff of this option at maturity is given by:

```math
\max\{S_T - K, 0\}
```

Our primary goal is to estimate the expected payoff, which directly represents the fair value of the option before discounting:

```math
\mathbb{E}\left[ \max\{S_T - K, 0\} \right]
```

Additionally, we seek to compute the sensitivity measure known as Delta ($\Delta$), defined as the derivative of the option's price with respect to the underlying spot price:

```math
\Delta = \mathbb{P}(S_T \geq K)
```

#### Uncertainty Model

To implement QMC, we first construct a quantum circuit encoding the underlying asset price uncertainty. Assuming the asset price $S_T$ follows a log-normal distribution, we discretize and truncate this distribution over an interval $[\text{low}, \text{high}]$, dividing it into $2^n$ grid points, where $n$ is the number of allocated qubits. The resulting quantum state encodes the probability distribution of prices as:

```math
|\psi\rangle = \sum_{i=0}^{2^n-1} \sqrt{p_i}|i\rangle,
```

where each basis state $|i\rangle$ corresponds to a discrete price level obtained through an affine transformation:

```math
i \mapsto \frac{\text{high} - \text{low}}{2^n - 1} i + \text{low}.
```

This quantum encoding enables efficient manipulation and subsequent payoff evaluations using amplitude estimation.

#### Payoff Function Implementation

The payoff function exhibits linearity above the strike price $K$ and is zero otherwise. This conditional payoff is implemented through:

1. A **quantum comparator** circuit, which sets an ancilla qubit if the spot price $S_T$ meets or exceeds the strike price $K$.
2. A controlled **piecewise linear approximation** to represent the linear payoff beyond the threshold. We exploit the approximation:

```math
\sin^2\left(\frac{\pi}{2} c_{\text{approx}} \left(x - \frac{1}{2}\right) + \frac{\pi}{4}\right) \approx \frac{\pi}{2} c_{\text{approx}} \left(x - \frac{1}{2}\right) + \frac{1}{2}, \quad \text{for small } c_{\text{approx}}.
```

Controlled quantum rotations effectively encode this approximation into quantum amplitudes, allowing precise quantum computations of the expected payoff.

#### Delta Evaluation

The Delta calculation is straightforward compared to the expected payoff. Since Delta measures the probability that the option expires in the money, it directly uses the comparator circuit's ancilla qubit. Thus, amplitude estimation is employed directly on the comparator's output without further approximation.

#### Example Quantum Implementation

A minimal implementation illustrating this approach using Qiskit is provided below:

```python
from qiskit_finance.applications.estimation import EuropeanCallDelta, LogNormalDistribution, LinearAmplitudeFunction
num_uncertainty_qubits = 3
# Parameters for the underlying asset's log-normal distribution
S, vol, r, T = 2.0, 0.4, 0.05, 40 / 365
mu = (r - 0.5 * vol**2) * T + np.log(S)
sigma = vol * np.sqrt(T)
mean = np.exp(mu + sigma**2 / 2)
stddev = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2))
low, high = max(0, mean - 3 * stddev), mean + 3 * stddev
# Quantum uncertainty model
uncertainty_model = LogNormalDistribution(
    num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high)
)
# Option parameters
strike_price, c_approx = 1.896, 0.25
breakpoints = [low, strike_price]
slopes, offsets = [0, 1], [0, 0]
f_min, f_max = 0, high - strike_price
# Quantum objective function for payoff approximation
european_call_objective = LinearAmplitudeFunction(
    num_uncertainty_qubits,
    slopes,
    offsets,
    domain=(low, high),
    image=(f_min, f_max),
    breakpoints=breakpoints,
    rescaling_factor=c_approx,
)
# Combine uncertainty and payoff models
num_qubits = european_call_objective.num_qubits
european_call_circuit = QuantumCircuit(num_qubits)
european_call_circuit.append(uncertainty_model, range(num_uncertainty_qubits))
european_call_circuit.append(european_call_objective, range(num_qubits))
# Direct Delta evaluation via amplitude estimation
european_call_delta = EuropeanCallDelta(
    num_state_qubits=num_uncertainty_qubits,
    strike_price=strike_price,
    bounds=(low, high),
    uncertainty_model=uncertainty_model,
)
```

This example highlights the seamless integration of quantum state preparation, amplitude estimation, and payoff evaluation, clearly illustrating the advantages offered by Quantum Monte Carlo in European option pricing.

[1] Quantum Risk Analysis. Woerner, S., & Egger, D. J. (2019). https://www.nature.com/articles/s41534-019-0130-6

[2] Option Pricing using Quantum Computers. Stamatopoulos, N., Egger, D. J., Sun, Y., Zoufal, C., Iten, R., Shen, N., & Woerner, S. (2020). https://quantum-journal.org/papers/q-2020-07-06-291/


### Quantum Monte Carlo for Asian Option Pricing


### Quantum Monte Carlo for Barrier Options Pricing

In this section, we apply Quantum Monte Carlo (QMC) techniques specifically to barrier basket options, characterized by a payoff function of the form:

```math
\max\{S_T^1 + S_T^2 - K, 0\}
```

where $ S_T^1 $ and $ S_T^2 $ denote the asset prices at maturity and $ K $ is the strike price. The core objective is to estimate the expected payoff:

```math
\mathbb{E}\left[\max\{S_T^1 + S_T^2 - K, 0\}\right]
```

by leveraging quantum amplitude estimation, a quantum algorithm capable of efficiently approximating expectations by analyzing probability distributions encoded into quantum states.

#### Uncertainty Model

The uncertainty in asset prices at maturity is modeled by preparing a quantum state encoding the joint probability distribution of asset prices $S_T^1$ and $S_T^2$. For simplicity, we assume that the underlying assets follow independent lognormal distributions, although correlated scenarios could also be accommodated with slight modifications.

Each asset’s price is discretized onto a finite grid defined by $2^{n_j}$ points within intervals $[\text{low}_j, \text{high}_j]$, where $n_j$ denotes the number of qubits assigned to asset $j$. Thus, the quantum state encoding both assets simultaneously becomes:

```math
|\psi\rangle = \sum_{i_1,i_2} \sqrt{p_{i_1,i_2}} |i_1\rangle |i_2\rangle,
```

where each computational basis state $|i_j\rangle$ maps back to the physical asset price via an affine transformation:

```math
i_j \mapsto \frac{\text{high}_j - \text{low}_j}{2^{n_j} - 1} i_j + \text{low}_j.
```

This encoding step is critical, accurately capturing the underlying statistical distributions required for reliable expectation estimation.

#### Payoff Function Implementation

To evaluate the payoff, we need to implement the sum $S_T^1 + S_T^2$ within a quantum circuit and check whether this sum exceeds the strike price $K$. This involves:

1. **Summation operator**: Using a quantum weighted adder, we calculate the sum of asset prices into an ancilla quantum register.
2. **Quantum comparator**: A comparator then tests whether this aggregated sum meets or exceeds the strike price $K$. If the threshold is met, it triggers an ancilla qubit state change, activating the payoff logic.

The payoff itself—linear beyond the strike threshold—is approximated using a controlled rotation on the quantum state. Specifically, we utilize a small-angle approximation based on trigonometric identities, efficiently mapping discrete quantum states to payoff values:

```math
\sin^2\left(\frac{\pi}{2} c_{\text{approx}} \left(x - \frac{1}{2}\right) + \frac{\pi}{4}\right) \approx \frac{\pi}{2} c_{\text{approx}} \left(x - \frac{1}{2}\right) + \frac{1}{2}, \quad \text{for small } |x - \frac{1}{2}|.
```

This approximation permits the linear portion of the payoff to be efficiently encoded into quantum amplitudes using controlled quantum rotations.

#### Quantum Circuit Example

Below is a succinct quantum circuit example illustrating the threshold comparison and payoff activation:

```python
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import WeightedAdder, IntegerComparator
# Define registers
num_qubits = [3, 3]  # Qubits per asset
sum_qubits = sum(num_qubits)  # Total qubits for sum
ancilla = AncillaRegister(1, "ancilla")  # Ancilla qubit for threshold check
qr_state = QuantumRegister(sum_qubits, "state")  # Asset state register
# Circuit initialization
qc = QuantumCircuit(qr_state, ancilla)
# Summation of S_T^1 and S_T^2
weights = [2**i for i in range(sum_qubits)]
weighted_adder = WeightedAdder(sum_qubits, weights)
qc.append(weighted_adder, qr_state)
# Comparison against strike price K
strike_price = 5  # Example strike price
comparator = IntegerComparator(sum_qubits, strike_price, geq=True)
qc.append(comparator, qr_state[:] + ancilla[:])
# Payoff activation (controlled rotation simplified example)
qc.cx(ancilla[0], qr_state[0])
qc.draw(output='mpl')
```

This illustrative example highlights the critical conditional logic distinguishing basket barrier options from other exotic types, capturing threshold conditions and linearly scaled payoffs within the quantum Monte Carlo framework.

### Quantum Monte Carlo for Digital Pricing

## Other Quantum Approaches for Exotic Option Pricing

### Option Pricing with qGANs

[1] Quantum Generative Adversarial Networks for Learning and Loading Random Distributions. Zoufal, C., Lucchi, A., & Woerner, S. (2019). https://www.nature.com/articles/s41534-019-0223-2



## Future Improvement

## Appendix

### Appendix A: Fundamentals of Quantum Computing for Option Pricing

This appendix provides a concise yet rigorous introduction to quantum computing concepts relevant to option pricing models. The primary advantage of quantum computing in finance lies in its ability to process large probability distributions efficiently, offering potential speedups for Monte Carlo simulations and differential equation solvers.

#### 1. Qubits and Superposition
A **qubit** (quantum bit) is the fundamental unit of quantum information. Unlike classical bits, which can be either $0$ or $1$, a qubit exists in a **superposition** of both states:

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