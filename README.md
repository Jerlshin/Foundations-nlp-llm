# Foundations of NLP & LLMs — From Scratch

> A modular, research-oriented framework for implementing Natural Language Processing and Large Language Models from first principles.

---

## 📌 Overview

This repository is a personal research and learning framework dedicated to understanding and implementing modern NLP and Large Language Model (LLM) systems **from scratch**, without relying on high-level pretrained abstractions.

The goal is not merely to use existing models, but to **reconstruct their foundations**, enabling deep conceptual mastery and experimental freedom.

This project emphasizes:

- Mathematical clarity
- Architectural transparency
- Modular design
- Reproducibility
- Research-oriented experimentation

---

## 🎯 Objectives

- Implement core NLP and LLM components from first principles
- Avoid opaque pretrained pipelines
- Understand internal mechanisms of Transformers and LLMs
- Enable plug-and-play experimentation
- Serve as a long-term research laboratory

---

## 🧠 Core Philosophy

> Understanding precedes optimization.

Every component in this repository is implemented with clarity and minimal abstraction.  
Each module is designed to be readable, testable, and interchangeable.

This framework prioritizes:

- Conceptual rigor over convenience
- Interpretability over shortcuts
- Foundations over frameworks

---

## 📂 Project Structure

```

foundations-nlp-llm/
│
├── tokenizers/          # Tokenization algorithms
├── embeddings/          # Word & positional embeddings
├── attention/           # Attention mechanisms
├── transformers/        # Transformer architectures
├── layers/              # Core neural layers
├── objectives/          # Training objectives
├── optimizers/          # Optimizers from scratch
├── schedulers/          # Learning rate schedulers
├── training/            # Training engines
├── evaluation/          # Metrics & benchmarks
├── inference/           # Decoding strategies
├── datasets/            # Raw & processed data
├── experiments/         # Reproducible experiments
├── notebooks/           # Analysis notebooks
├── scripts/             # CLI utilities
├── docs/                # Theory & notes
└── utils/               # Shared utilities

````

Each directory contains independent, composable modules.

---

## 🧩 Implemented Components

### Tokenization
- Word-level
- Byte Pair Encoding (BPE)
- Unigram Language Model
- SentencePiece
- Byte-level BPE
- Neural tokenizers (experimental)

### Embeddings
- Learned embeddings
- Word2Vec
- GloVe
- FastText
- Sinusoidal positional encoding
- Learned positional encoding
- RoPE
- ALiBi

### Attention Mechanisms
- Dot-product attention
- Scaled dot-product attention
- Multi-head attention
- Rotary attention
- Sparse attention
- Flash attention

### Transformer Architectures
- Encoder-only (BERT-style)
- Decoder-only (GPT-style)
- Encoder–Decoder (T5-style)
- LLaMA-style variants

### Optimization
- SGD
- Adam
- AdamW
- Lion
- Adafactor

### Training Objectives
- Cross-entropy
- Masked Language Modeling (MLM)
- Causal Language Modeling (CLM)
- Contrastive learning
- Natural Language Inference (NLI)

---

## ⚙️ Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy
- Datasets
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
````

---

## 🚀 Getting Started

### Example: Training a Transformer on AG News

```bash
python scripts/train.py \
    --model transformer \
    --dataset ag_news \
    --tokenizer unigram \
    --epochs 10
```

### Example: Running Inference

```bash
python scripts/inference.py \
    --checkpoint checkpoints/model.pt \
    --text "Artificial intelligence is transforming research."
```

---

## 🧪 Experiments

All reproducible experiments are stored under:

```
experiments/
```

Each experiment contains:

* Configuration files
* Training logs
* Model checkpoints
* Evaluation results

This enables systematic ablation studies and comparisons.

---

## 📊 Evaluation

Supported evaluation metrics include:

* Accuracy
* Perplexity
* BLEU
* ROUGE
* GLUE benchmarks
* NLI benchmarks

Evaluation scripts are located in:

```
evaluation/
```

---

## 📓 Notebooks

Interactive research notebooks are located in:

```
notebooks/
```

These include:

* Tokenization analysis
* Attention visualization
* Scaling law studies
* Ablation experiments

---

## 📖 Documentation

Theoretical explanations and research notes are available in:

```
docs/
```

Topics include:

* Transformer mathematics
* Tokenization theory
* Attention mechanisms
* Optimization dynamics
* Scaling laws

---

## 🔬 Research Orientation

This repository is designed to support:

* Rapid prototyping
* Architecture ablations
* Tokenization experiments
* Optimization studies
* Interpretability research

It can serve as a foundation for:

* Academic research
* Thesis work
* Open-source contributions
* Advanced experimentation

---

## 📈 Roadmap

### Phase 1 — Foundations

* Core Transformer
* Basic tokenizers
* Training loop
* Evaluation pipeline

### Phase 2 — Scaling

* Efficient attention
* Distributed training
* Mixed precision
* Large corpora

### Phase 3 — Advanced Models

* LLaMA-style models
* Multimodal models
* Retrieval-augmented generation
* Agentic systems

---

## 🤝 Contributing

This is currently a personal research project.

Contributions, discussions, and suggestions are welcome through issues and pull requests.

All code should adhere to:

* Clarity
* Documentation
* Reproducibility
* Minimal abstraction

---

## 📜 License

This project is released under the MIT License.

See `LICENSE` for details.

---

## 🙏 Acknowledgements

Inspired by foundational work from:

* Vaswani et al. (2017)
* Devlin et al. (2018)
* Brown et al. (2020)
* Touvron et al. (2023)
* EleutherAI
* HuggingFace
* OpenAI research

---

## ✝️ Personal Note

> “By wisdom a house is built, and by understanding it is established.”

This repository is built as a long-term pursuit of understanding, rigor, and responsible research.

---

## 📬 Contact

Maintained by: **Jerlshin**

For academic collaboration and research discussion, feel free to reach out via GitHub.

---

