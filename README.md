# 🧠 QMind v2.0 — Quantum-Inspired AI Reasoning System

> *A self-evolving, symbolic AI reasoning engine that applies quantum mechanics principles — superposition, interference, and wavefunction collapse — entirely on a normal computer. No quantum hardware needed.*

Built by **S. Vinay Narasimha** · MIT License · Python 3 · 100% Offline

---

## 📖 What Is This Project?

QMind is an AI reasoning system that borrows ideas from quantum physics to think differently from regular AI.

Instead of following one reasoning path and giving you one answer, QMind explores **many possible answers simultaneously** (like quantum superposition), lets strong answers **reinforce each other** (constructive interference), lets contradicting answers **cancel each other out** (destructive interference), and then **collapses** to the best answer — exactly like a quantum wavefunction being measured.

It stores knowledge in a **graph** (a web of connected concepts), reasons using **8 different logic strategies**, has **5 tiers of memory**, knows when it does not know something, and can even **invent new concepts** by spotting patterns in what it already knows.

All of this runs on your regular laptop or PC. No cloud, no API, no internet needed.

---

## 🖥️ System Requirements

### Minimum (to just run and try it)
| Component | Minimum |
|-----------|---------|
| RAM | **512 MB free** |
| Python | 3.9 or higher |
| Disk | ~200 MB (for libraries) |
| CPU | Any dual-core (2010 or newer) |
| OS | Windows / macOS / Linux |

### Recommended (comfortable experience)
| Component | Recommended |
|-----------|-------------|
| RAM | **2 GB free** |
| Python | 3.10 or 3.11 |
| CPU | Quad-core (2015 or newer) |

### Default settings RAM usage (what the code uses out of the box)
| Part of the system | RAM used |
|---|---|
| Knowledge graph (20 demo concepts) | ~5 MB |
| Working memory (20 concepts max) | ~2 MB |
| Episodic memory (500 sessions max) | ~30 MB |
| Cold cache (2,000 concepts max) | ~50 MB |
| Embedding vectors (64 dimensions) | ~10 MB |
| Library overhead (networkx, numpy, qutip) | ~200 MB |
| **Total (default settings)** | **~300 MB** |

> 💡 **So in practice:** If your computer has 1 GB of free RAM, QMind will run just fine at default settings.

---

## 🎛️ How to Reduce RAM by Adjusting the Settings

QMind has a configuration block at the very top of `Qmind_v2.py` called `CFG`. This is where all the numbers that control memory usage live. You do not need to understand the whole code — just change these numbers.

Open `Qmind_v2.py` and find this section near the top:

```python
CFG = {
    "working_memory_capacity":  20,
    "episodic_memory_capacity": 500,
    "cold_cache_capacity":      2000,
    "complexity_budget":        400,
    "embed_dim":                64,
    ...
}
```

Here is what each one does and how to reduce it:

---

### `working_memory_capacity` — *How many concepts the system holds in active thought*

Think of this like the number of browser tabs you keep open. The default is **20**.

| Your RAM situation | Set it to | Effect |
|---|---|---|
| Very low RAM (< 512 MB free) | `5` | Uses ~0.5 MB |
| Low RAM (512 MB – 1 GB) | `10` | Uses ~1 MB |
| Normal (1 – 2 GB) | `20` *(default)* | Uses ~2 MB |
| Plenty of RAM (2 GB+) | `30–50` | More concepts in focus |

```python
"working_memory_capacity": 10,   # ← change this number
```

---

### `episodic_memory_capacity` — *How many past reasoning sessions to remember*

This is like a diary of everything QMind has thought about. The default is **500** sessions.

| Your RAM situation | Set it to | RAM saved |
|---|---|---|
| Very low RAM | `50` | Saves ~27 MB |
| Low RAM | `100` | Saves ~24 MB |
| Normal | `500` *(default)* | — |
| Plenty of RAM | `1000–2000` | Better long-term learning |

```python
"episodic_memory_capacity": 100,   # ← change this number
```

---

### `cold_cache_capacity` — *How many evicted concepts to keep in standby*

When a concept leaves active working memory, it goes here instead of being deleted completely — so it can be recalled quickly. The default is **2,000**.

| Your RAM situation | Set it to | RAM saved |
|---|---|---|
| Very low RAM | `100` | Saves ~45 MB |
| Low RAM | `500` | Saves ~37 MB |
| Normal | `2000` *(default)* | — |
| Plenty of RAM | `5000+` | Faster recall of old concepts |

```python
"cold_cache_capacity": 500,   # ← change this number
```

---

### `complexity_budget` — *How many reasoning paths to explore per question*

This is how hard QMind "thinks" for each answer. The default is **400 paths**.

This setting affects **CPU time and RAM together** — higher = smarter answers but slower and more memory used.

| Situation | Set it to | Effect |
|---|---|---|
| Very slow / low RAM machine | `50–100` | Fast answers, less thorough |
| Normal machine | `400` *(default)* | Balanced |
| Powerful machine | `800–1500` | Deeper, more thorough reasoning |

```python
"complexity_budget": 100,   # ← change this number
```

---

---

### ⚠️ THE MOST IMPORTANT SETTING — `MAX_QUBITS` inside the `reason()` method

This is the single biggest RAM decision in the entire project, and it is **not in the CFG block** — it is hardcoded inside the `reason()` method. Search for this line in `Qmind_v2.py`:

```python
MAX_QUBITS = 18
```

You will find it around line 1932.

**Here is what this number actually controls and why it is so critical:**

Every time QMind answers a question, it builds a **quantum state vector** — a list of complex numbers where each position represents one possible combination of concepts being "true" at the same time. This is the superposition state. The length of this vector is **2 raised to the power of n_qubits**.

So if `MAX_QUBITS = 18`, the state vector has `2^18 = 262,144` positions.

Each position holds one `complex128` number, which takes **16 bytes** of RAM.

**This is how the RAM scales with MAX_QUBITS:**

| MAX_QUBITS | State vector size (2^n) | RAM for state vector | Concepts processed | Minimum free RAM needed |
|---|---|---|---|---|
| 10 | 1,024 | ~0.016 MB | 10 concepts | ~300 MB |
| 12 | 4,096 | ~0.064 MB | 12 concepts | ~350 MB |
| 14 | 16,384 | ~0.26 MB | 14 concepts | ~400 MB |
| 16 | 65,536 | ~1 MB | 16 concepts | ~600 MB |
| **18** | **262,144** | **~4 MB** | **18 concepts** | **~800 MB** ← default |
| 20 | 1,048,576 | ~16 MB | 20 concepts | ~4 GB |
| 22 | 4,194,304 | ~64 MB | 22 concepts | ~8 GB |
| 24 | 16,777,216 | ~256 MB | 24 concepts | ~16 GB |
| 26 | 67,108,864 | ~1 GB | 26 concepts | ~32 GB |

> 🚨 **This is why going above 2^20 needs 16 GB+ of RAM.** The state vector alone at `MAX_QUBITS = 24` takes 256 MB, and this is allocated **fresh every single time you ask a question.** On top of everything else the system already holds in memory (libraries, graph, episodic memory), you very quickly run out of RAM. At `MAX_QUBITS = 26`, the state vector is 1 GB — allocated and deallocated on every reasoning call.

**The key thing to understand:** This is not just storage — Python allocates this entire array **during every single call to `reason()`**. So if you ask 10 questions in parallel, 10 of these arrays exist in RAM simultaneously.

---

**How to change it — find this block in `Qmind_v2.py`:**

```python
MAX_QUBITS = 18
if len(all_effective_nodes) > MAX_QUBITS:
    # Keep start_concept + highest-salience nodes
    sal_sorted = sorted(
        all_effective_nodes,
        ...
    )[:MAX_QUBITS]
```

Change only the number `18`. Always use a whole number. **Do not go above 20 unless you have 16 GB+ of free RAM.**

**Recommended values by available RAM:**

| Free RAM available | Set MAX_QUBITS to | State vector RAM | Reasoning quality |
|---|---|---|---|
| Less than 512 MB | `10` | ~0.016 MB | Basic — fewer concepts considered |
| 512 MB – 1 GB | `14` | ~0.26 MB | Decent — good for most questions |
| 1 GB – 2 GB | `16` | ~1 MB | Good — handles most reasoning well |
| 2 GB – 4 GB | `18` ← default | ~4 MB | Great — full system capability |
| 4 GB – 8 GB | `20` | ~16 MB | Excellent — very deep reasoning |
| 8 GB – 16 GB | `22` | ~64 MB | Powerful — research-grade depth |
| 16 GB+ | `24` | ~256 MB | Maximum — only for specialised use |

**Why does more qubits = better reasoning?**

Each qubit represents one concept. With 18 qubits, QMind can represent all possible combinations of up to 18 concepts being simultaneously "active" in the superposition. This means it can detect more complex interference patterns — for example, noticing that concepts A, B, and C together strongly imply D, even if no single one of them does alone. Fewer qubits means simpler, shallower combinations.

---

### `embed_dim` — *The size of each concept's mathematical fingerprint (2^n)*

This is the one setting that works on the **power of 2 (2^n)** principle. Every concept in QMind gets converted into a list of numbers (called an embedding vector) so the system can measure how similar two concepts are. The default is **64**, which equals 2^6.

**This is how 2^n works here:**

| n | Value (2^n) | RAM per 1,000 concepts | Accuracy |
|---|---|---|---|
| 4 | **16** | ~0.1 MB | Very rough similarity |
| 5 | **32** | ~0.2 MB | Basic similarity |
| **6** | **64** *(default)* | **~0.5 MB** | **Good balance** |
| 7 | **128** | ~1 MB | Better similarity |
| 8 | **256** | ~2 MB | Very accurate similarity |
| 9 | **512** | ~4 MB | Near-perfect (overkill for most uses) |

**To change it:**
```python
"embed_dim": 32,   # ← use a power of 2: 16, 32, 64, 128, 256
```

> ⚠️ **Important:** Always use a power of 2 for this value (16, 32, 64, 128, 256...). This is because the random projection matrix inside the code is designed around powers of 2 for mathematical stability. Using other numbers like 50 or 100 will technically work but may give less reliable similarity scores.

---

### Ultra Low RAM Config (for machines with less than 512 MB free)

Copy and paste this entire CFG block to replace the default one:

```python
CFG = {
    "graph_file":               "qmind_graph.graphml",
    "episodic_file":            "qmind_episodic.json",
    "unknown_log":              "qmind_unknown.txt",

    "working_memory_capacity":  5,      # was 20
    "episodic_memory_capacity": 50,     # was 500
    "cold_cache_capacity":      100,    # was 2000

    "default_depth":            3,      # was 4
    "complexity_budget":        80,     # was 400
    "min_amplitude":            0.01,   # was 0.003
    "noise_scale":              0.004,

    "base_learning_rate":       0.08,
    "reinforcement_factor":     0.12,
    "decay_factor":             0.92,
    "decay_threshold_s":        600,

    "emergence_min_shared":     2,
    "emergence_min_conf":       0.55,
    "curiosity_gap_threshold":  0.35,
    "curiosity_contradiction":  0.4,

    "embed_dim":                32,     # was 64 (power of 2)
    "collapse_similarity_th":   0.72,
}
```

**Expected total RAM: ~120–150 MB.** QMind will still work — it will just think with a smaller memory and explore fewer paths per question.

---

### High Performance Config (for machines with 4 GB+ free RAM)

```python
CFG = {
    "graph_file":               "qmind_graph.graphml",
    "episodic_file":            "qmind_episodic.json",
    "unknown_log":              "qmind_unknown.txt",

    "working_memory_capacity":  50,     # was 20
    "episodic_memory_capacity": 2000,   # was 500
    "cold_cache_capacity":      10000,  # was 2000

    "default_depth":            5,      # was 4
    "complexity_budget":        1200,   # was 400
    "min_amplitude":            0.001,  # was 0.003
    "noise_scale":              0.004,

    "base_learning_rate":       0.08,
    "reinforcement_factor":     0.12,
    "decay_factor":             0.92,
    "decay_threshold_s":        600,

    "emergence_min_shared":     2,
    "emergence_min_conf":       0.55,
    "curiosity_gap_threshold":  0.35,
    "curiosity_contradiction":  0.4,

    "embed_dim":                128,    # was 64 (power of 2)
    "collapse_similarity_th":   0.72,
}
```

**Expected total RAM: ~800 MB – 1.2 GB.** Much deeper reasoning, better long-term memory, and more accurate semantic similarity.

---

## 📦 Libraries Used

QMind uses 5 external libraries. All of them are free and open source.

### 1. `networkx`
**What it does:** Stores and manages the knowledge graph — the web of concepts and their relationships that QMind reasons over. Handles path-finding, graph traversal, PageRank computation, and GraphML file saving/loading.

**Why this library:** NetworkX is the most complete graph library in Python. It supports directed graphs, rich edge metadata, all the path algorithms QMind needs, and standard file formats.

**Docs:** https://networkx.org

---

### 2. `numpy`
**What it does:** Powers all the maths in QMind — amplitude vectors, complex number arithmetic, random projection matrices for embeddings, statistical calculations, and array operations throughout the codebase.

**Why this library:** NumPy is the foundation of scientific computing in Python. It is fast, reliable, and required by almost every other library here.

**Docs:** https://numpy.org

---

### 3. `scipy`
**What it does:** Provides advanced scientific computing utilities. Used in QMind for signal processing functions and statistical operations that extend what NumPy can do.

**Why this library:** SciPy complements NumPy with higher-level scientific algorithms. It is also a dependency of qutip.

**Docs:** https://scipy.org

---

### 4. `scikit-learn`
**What it does:** Used specifically for `cosine_similarity` — the mathematical function that measures how similar two concept embedding vectors are. This is what allows QMind to know that "Gene" and "DNA" are more related than "Gene" and "Automobile".

**Why this library:** scikit-learn's cosine similarity is numerically stable, well-tested, and works perfectly with NumPy arrays. Only one function from the library is used, but it is a critical one.

**Docs:** https://scikit-learn.org

---

### 5. `qutip`
**What it does:** QuTiP (Quantum Toolbox in Python) is a real quantum physics library. QMind uses it to represent quantum state objects (`Qobj`) — giving the amplitude engine a mathematically proper quantum representation rather than just mimicking the numbers.

**Why this library:** This is what makes QMind genuinely "quantum-inspired" rather than just calling itself that. The quantum state objects follow real quantum mechanical rules.

**Docs:** https://qutip.org

---

## ⚙️ Installation

### Step 1 — Make sure you have Python 3.9 or newer

Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and type:

```bash
python --version
```

You should see something like `Python 3.10.12`. If you get an error or see Python 2.x, download Python from https://python.org/downloads

---

### Step 2 — (Recommended) Create a virtual environment

A virtual environment keeps QMind's libraries separate from the rest of your system. This is optional but strongly recommended.

```bash
# Create the environment
python -m venv qmind_env

# Activate it — on Windows:
qmind_env\Scripts\activate

# Activate it — on Mac / Linux:
source qmind_env/bin/activate
```

You will know it is activated when you see `(qmind_env)` at the start of your terminal line.

---

### Step 3 — Install all libraries in one command

```bash
pip install networkx numpy scipy scikit-learn qutip
```

That's it. This will download and install all 5 libraries. It may take 2–5 minutes depending on your internet speed.

**Expected sizes:**
| Library | Download size |
|---|---|
| networkx | ~2 MB |
| numpy | ~15 MB |
| scipy | ~35 MB |
| scikit-learn | ~8 MB |
| qutip | ~15 MB |
| **Total** | **~75 MB** |

---

### Step 4 — Run QMind

```bash
python Qmind_v2.py
```

The first time you run it, QMind will:
1. Build the demo knowledge graph (20 concepts, 24 connections)
2. Save it to `qmind_graph.graphml` in the same folder
3. Run 6 demonstration queries and print full reasoning traces
4. Run parallel reasoning, temporal simulation, counterfactual analysis, and more
5. Save its memory to `qmind_episodic.json`

---

### Step 5 — Use it in your own Python code

```python
from Qmind_v2 import Qmind, CognitiveGoal

# Start the system
qmind = Qmind(graph_file="qmind_graph.graphml", episodic_file="qmind_episodic.json")

# Ask a question
result = qmind.answer("What is DNA?")
print(result["answer"])

# Ask with a specific reasoning goal
result = qmind.answer("What is Consciousness?", goal=CognitiveGoal.EXPLORE)
print(result["reasoning_trace"].summary())

# Add your own knowledge
qmind.graph_manager.add_concept("Machine Learning", confidence=0.95, domain="computer_science", info="Algorithms that learn from data.")
qmind.graph_manager.add_edge("Machine Learning", "Intelligence", "contributes_to", weight=0.9, confidence=0.9)

# Ask about what you just taught it
result = qmind.answer("What is Machine Learning?")
print(result["answer"])
```

---

## 🗂️ Files Created When QMind Runs

| File | What it is |
|---|---|
| `qmind_graph.graphml` | The knowledge graph — all concepts and connections, saved between runs |
| `qmind_episodic.json` | Memory of past reasoning sessions — QMind learns from these |
| `qmind_unknown.txt` | Log of concepts QMind was asked about but did not recognise |

These files are created automatically in the same folder as `Qmind_v2.py`. You can delete them to reset QMind to a fresh state.

---

## 🚀 Features at a Glance

- **15 cognitive subsystems** working together as one reasoning engine
- **8 reasoning modes** — deductive, inductive, abductive, analogical, causal, temporal, counterfactual, meta
- **5-tier memory** — working → episodic → semantic → long-term → cold cache
- **Quantum amplitude engine** — real superposition, interference, and wavefunction collapse mathematics
- **Curiosity engine** — finds its own knowledge gaps and generates questions
- **Emergent synthesis** — invents new concepts by spotting patterns
- **Meta-cognition** — monitors and regulates the quality of its own reasoning
- **Contradiction manager** — handles conflicting beliefs without crashing
- **Fully offline** — no internet, no API keys, no cloud
- **Deterministic by default** — same question = same answer every time (seed=42)
- **Parallel reasoning** — can reason about multiple concepts simultaneously

---

## 📄 License

MIT License — free to use, modify, distribute, and build on. See `LICENSE` file.

---

## 👤 Author

**S. Vinay Narasimha**

Developed with the assistance of Claude (Anthropic), Gemini (Google), and ChatGPT (OpenAI).

---

*"What if an AI could think the way a quantum computer computes — holding all possibilities at once, and collapsing to the best answer?"*
