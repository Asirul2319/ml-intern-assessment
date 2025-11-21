# AI/ML Intern Assignment

### Trigram Language Model (Task 1)

### Scaled Dot-Product Attention (Task 2 – Optional)

This repository contains two major components required for the AI/ML Intern Assessment:

1. **A Trigram Language Model built completely from scratch**
2. **A NumPy-only implementation of the Scaled Dot-Product Attention mechanism**

Both tasks are implemented using simple, readable Python code with minimal dependencies. The goal was to demonstrate understanding of classical n-gram language modeling and the core mathematical operation behind Transformer architectures.

---

# Folder Structure

```
ml-assignment/
│
├── src/
│   ├── ngram_model.py            # Trigram Model implementation (Task 1)
│   ├── generate.py               # Script to train + generate text
│   └── utils.py                  # Optional helpers
│
├── task2/
│   ├── attention.py              # Scaled Dot-Product Attention (Task 2)
│   └── demo_attention.py         # Demonstration for Task 2
│
├── data/                         # Training text files
│
├── tests/                        # Pytest cases for Task 1
│
└── evaluation.md                 # Design explanation for both tasks
```

---

#  Task 1 — Trigram Language Model

## Overview

The trigram model learns the probability distribution of words based on their preceding two-word context. It captures local language patterns from a dataset and generates new text by sampling from these learned probabilities.

---

## How It Works (Code-Level Summary)

### **1. Text Cleaning & Tokenization**

* Uses a small regex to keep only alphanumeric words and apostrophes.
* Converts text to lowercase.
* Splits tokens into a list.

### **2. Padding**

Special tokens are added:

* `<s> <s>` at the beginning
* `</s>` at the end

This ensures we can form valid trigrams for the first and last words.

### **3. Storing N-gram Counts**

The model uses:

* `defaultdict(Counter)` for trigram counts
* `Counter` for unigrams

Example internal structure:

```
tri[(w1, w2)][w3] = count
```

### **4. Text Generation**

* Starts with context `<s>, <s>`
* Uses `random.choices` to sample the next word based on trigram count weights
* Falls back to unigrams if context not found

This method produces natural and varied sentences.

---

## How to Run Task 1

### 1. Navigate to the project root

```
cd ml-assignment
```

### 2. Add a training text file to the `data/` directory

Example:

```
data/Alice_Adventures_in_Wonderland_by_Lewis_Carroll.txt
```

### 3. Update the file path inside `src/generate.py`

```python
with open("data/your_file.txt", "r", encoding="utf-8") as f:
```

### 4. Run the generator script

```
python src/generate.py
```

### Example Output

```
i haven't the slightest idea said the mouse with an important air
```

---

# Task 2 — Scaled Dot-Product Attention (Optional)

## Overview

This implementation recreates the core attention formula used in Transformer models such as BERT and GPT. The function is implemented with **only NumPy**.

The formula:

```
Attention(Q, K, V) = softmax((QKᵀ) / sqrt(dk)) V
```

---

## How It Works

### **1. Input Conversion and Validation**

* Q, K, V are converted to NumPy arrays.
* Shapes are validated (Q and K depth must match).

### **2. Score Calculation**

```
scores = (Q @ K.T) / sqrt(dk)
```

### **3. Mask Application**

* Supports boolean masks and 0/1 masks.
* Broadcasts automatically.
* Masked positions replaced with `-1e9` before softmax.

### **4. Softmax + Weighted Sum**

* Stable softmax applied along last axis.
* Multiply weights with V to get final output.

---

## How to Run Task 2

### 1. Navigate to the Task 2 directory

```
cd ml-assignment/task2
```

### 2. Run the demo script

```
python demo_attention.py
```

### Example Output

```
Weights:
[[0.39 0.21 0.39],
 [0.21 0.39 0.39]]

Output:
[[41.62  0.39],
 [43.16  0.39]]
```

Masked results will also appear to verify correctness.

---

# Running Tests (Task 1)

To run all tests inside the `tests/` directory:

```
pytest -q
```

You should see:

```
3 passed in X.XXs
```

---

# Design Documentation

A complete explanation of the design decisions for both Task 1 and Task 2 is provided in:

```
evaluation.md
```

This includes:

* n-gram counting logic
* cleaning & padding strategy
* unknown word handling
* probabilistic generation
* attention scaling, masking, and softmax behavior

---

# Summary

This project demonstrates:

* An end-to-end trigram language model built from raw text
* Strong understanding of probabilistic text generation
* Correct implementation of the attention mechanism using NumPy
* Clean, simple, student-friendly code following assignment rules

Both tasks are complete, tested, and ready for review.
