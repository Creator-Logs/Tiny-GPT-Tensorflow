# 🧠 Shakespeare Transformer Decoder (TensorFlow)

This project recreates [Andrej Karpathy’s NanoGPT](https://github.com/karpathy/nanoGPT) — a simplified GPT-2 architecture — **implemented entirely in TensorFlow** instead of PyTorch.
It demonstrates how to build, train, and generate text using a **Transformer Decoder** model on the complete works of Shakespeare (and optionally, larger datasets).

---

## 📘 Overview

The goal of this project is to:

* Recreate the **GPT-2 architecture (decoder-only Transformer)** with fewer parameters
* Implement it in **TensorFlow**, showcasing a framework-agnostic approach
* Train it on **Shakespeare’s text** and optionally **larger corpora** to observe scaling behavior
* Generate coherent text using learned token probabilities

---

## 🧩 Features

* ✅ Pure **TensorFlow 2.x** implementation — no PyTorch required
* 🧱 **Custom-built Transformer Decoder** (multi-head attention, feed-forward layers, layer norm, etc.)
* 📊 **Configurable hyperparameters** for model size, layers, embedding dimensions, etc.
* 🧠 **Text generation** using temperature sampling
* 📚 **Training on larger datasets** for extended generalization
* 💾 **Checkpoint saving** and **resume training** support

---

## 📂 Project Structure

```
├── main.ipynb      # Main notebook (model, training, and generation)
└── data/           # Dataset directory (contains shakespeare.txt or other corpora)
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Creator-Logs/Tiny-GPT-Tensorflow.git
cd Shakespeare-Transformer-Decoder
```

### 2. Install Dependencies

```bash
pip install tensorflow matplotlib
```

### 3. Run the Notebook

Open the main notebook in Jupyter or Google Colab/Kaggle:

```bash
jupyter notebook notebook.ipynb
```

Or upload it to [Google Colab](https://colab.research.google.com/) or [Kaggle](https://www.kaggle.com/) and run all cells.

You can also just run in on VScode
---

## 🧠 Model Architecture

The Transformer Decoder is built from:

* Token + Positional Embeddings
* Multi-Head Self-Attention
* Feed-Forward Network
* Layer Normalization
* Residual Connections

It’s a smaller, TensorFlow-native version of GPT-2, ideal for experimentation and educational purposes.

---

## 📈 Training

You can train on Shakespeare by default or replace the dataset with any large text corpus. Training automatically tokenizes the input text, creates sequences, and feeds batches into the model.

---

## 🔬 Future Improvements

* [ ] Add BPE tokenization
* [ ] Experiment with mixed-precision training
* [ ] Implement fine-tuning on modern datasets
* [ ] Convert model to TensorFlow Lite for deployment

---

## 🙌 Acknowledgments

* Inspired by [Andrej Karpathy’s NanoGPT](https://github.com/karpathy/nanoGPT)
* Built with TensorFlow

---

## 🧑‍💻 Author

**Ansh Gupta**

Engineer (I build stuff)

---
