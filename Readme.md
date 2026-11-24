# Perplexity Explorer — How Language Models Measure Uncertainty

An interactive, first-principles web application exploring how causal language models assign probabilities to text, quantify surprise through cross-entropy loss, and condense sequential uncertainty into **perplexity**.

Powered by **DistilGPT-2 (82M parameters)** running client-side in the browser via **Transformers.js** and **ONNX WebAssembly**, styled with the editorial design language of *LLM Pipeline Explorer*.

---

## 🧭 Live Features & Architecture

### 1. Live Model Explorer & Surprise Heatmap
- **In-Browser DistilGPT-2 Inference**: Computes exact log-probabilities for every token using `@huggingface/transformers` (`Xenova/distilgpt2`) with quantized `q8` ONNX weights.
- **Valid Conditional Prior**: Prepends the GPT-2 document boundary token `<|endoftext|>` (`ID: 50256`) so the first token is assigned a genuine conditional prior probability $P(t_1 \mid \langle|\text{endoftext}|\rangle)$.
- **Interactive Heatmap Chips**: Color-codes every token by its negative log-likelihood (nats):
  - 🟢 **Low Surprise** ($P \ge 30\%$): highly expected tokens.
  - 🟡 **Expected** ($10\% \le P < 30\%$): standard transitions.
  - 🟠 **Surprising** ($1\% \le P < 10\%$): uncommon syntax or rare words.
  - 🟣 **Anomaly** ($P < 1\%$): unexpected or grammatically anomalous tokens.
- **Deep Token Inspector**: Click any token to inspect its vocabulary rank (e.g. `#1 of 50,257`), prior context string, exact reciprocal surprise ($1/P$), and the **Top 5 candidate alternatives** DistilGPT-2 weighed at that exact position.
- **Dual-Metric Plotly Visualization**: Interactive bar & line chart plotting token-by-token surprise (nats) and predicted probability with responsive light/dark theme adaptation.

### 2. Next-Token Horizons & Interactive Temperature
- Inspect the top 10 tokens DistilGPT-2 predicts will come immediately after your input.
- **Dynamic Temperature Slider ($T \in [0.1, 2.0]$)**:
  $$\tilde{P}(w) = \frac{\exp(\text{logit}_w / T)}{\sum_{v} \exp(\text{logit}_v / T)}$$
  Watch probability mass concentrate toward greedy peaks at $T=0.1$ or flatten toward uniform maximum entropy at $T=2.0$.
- **Interactive Sentence Builder**: Click `+ Append` on any predicted candidate to extend your text token-by-token and observe how perplexity shifts in real time.

### 3. Sentence Contrast Mode (A vs B)
- Directly compare two sentences side-by-side:
  - *Natural Order vs Inverted*: `"The chef cooked a delicious meal."` vs `"The meal cooked a delicious chef."`
  - *Grammar / Fluency*: `"I love writing code in Python."` vs `"Python writing love I code in."`
  - *Semantic Common Sense*: `"The doctor examined the sick patient."` vs `"The sick patient examined the doctor."`
- Displays comparative perplexity scores, percent ratios, and highlights where syntax broke down.

### 4. The Perplexity Sandbox ("Try it by Hand")
- Drag probability sliders by hand to simulate arbitrary token sequences without needing neural network forward passes.
- Instant pre-configured presets:
  - **Uniform 8-sided Die**: $P = 0.125 \implies \text{PPL} = 8.00$
  - **Coin Flip Sequence**: $P = 0.50 \implies \text{PPL} = 2.00$
  - **High Certainty**: $P \approx [0.85, 0.90, 0.88] \implies \text{PPL} \approx 1.14$
  - **The "Black Swan" Trap**: Demonstrates how a single token with $P = 0.005$ skyrockets sequence perplexity!

---

## 📐 Mathematical Foundations

### The Geometric Mean of Surprises
Perplexity can be understood as the geometric mean of the reciprocal probabilities assigned to each token:

$$\text{Perplexity} = \left( \prod_{i=1}^N \frac{1}{P(t_i \mid t_{<i})} \right)^{1/N}$$

- $1/P(t_i)$ represents how many equally likely outcomes would have yielded that token ("1 in $X$ chance").
- Taking the $N$-th root normalizes by sequence length, giving the **effective branching factor**.

### The Log-Space Formulation
To prevent computer floating-point arithmetic underflow when multiplying small fractions, the calculation is performed in log-space:

$$\text{Loss } \mathcal{L} = \frac{1}{N} \sum_{i=1}^N -\ln P(t_i \mid t_{<i})$$

$$\text{Perplexity} = \exp(\mathcal{L})$$

$$\mathcal{L} \text{ in bits/token} = \frac{\mathcal{L} \text{ in nats}}{\ln(2)}$$

A perplexity of **$K$** means that on average, the language model was as uncertain as choosing uniformly between **$K$ equally likely words** at each step.

---

## 🎨 Design System & UI/UX

Inspired by the editorial elegance of *LLM Pipeline Explorer*:
- **Typography**: Newsreader (editorial serif headings), Plus Jakarta Sans (body), and JetBrains Mono (tokens and numerical data).
- **Warm Paper Palette**: `#fcfaf7` background with soft terracotta accent (`#c25e2e` light / `#e07a4b` dark) and cream cards (`#ffffff`).
- **Dark Mode**: High-contrast, warm dark theme (`#161618`) with toggle and system preference detection.
- **Zero-Latency Initial Load**: Bundles instant precomputed data for standard presets so the interface is fully populated on first open before downloading the 62MB ONNX model.

---

## 🚀 How to Run Locally

Because the application uses ES modules and Web Workers for ONNX runtime, run it from any local HTTP server:

```bash
# Using Python 3
python3 -m http.server 8000

# Or using Node.js
npx serve .
```

Then open your browser to:
```
http://localhost:8000/Index.html
```

---

## 📦 Project Structure

```
perplexity explorer/
├── Index.html     # Self-contained web application (HTML, CSS, JS, ONNX & Plotly integration)
└── Readme.md      # Documentation and mathematical reference
```
