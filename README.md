# Impact of Kernel Size & Stride on CNN Performance

**Deep Learning Project | Amrita Vishwa Vidyapeetham, Coimbatore**

Amitha Mahesh · Meghana Reddy M · Nitheka A · Shivvanii S

---

## What This Project Is About

We wanted to understand what actually affects CNN performance — kernel size or stride?

Instead of using PyTorch or TensorFlow, we built a CNN from scratch using only NumPy. We tested 9 different configurations on the MNIST dataset and measured accuracy, prediction confidence, compute cost, and training time.

---

## What We Tested

- **Kernel sizes:** 3×3, 5×5, 7×7
- **Strides:** 1, 2, 3
- **Total configurations:** 9 (each trained independently for 10 epochs)

---

## Dataset

MNIST handwritten digit dataset (28×28 grayscale images)

| Split      | Samples |
|------------|---------|
| Training   | 2,000   |
| Validation | 1,000   |
| Test       | 1,000   |

---

## CNN Architecture (Built from Scratch in NumPy)

```
Conv2D → ReLU → MaxPool2D → Flatten → Dense(128) → ReLU → Dense(10) → Softmax
```

- **8 filters**, variable kernel size and stride
- **Mini-batch SGD**, batch size 32, learning rate 0.01
- **Cross-entropy loss**, 10 epochs
- Backpropagation manually implemented for all layers

---

## Results

| Kernel | Stride | Val Accuracy | KL Divergence | FLOPs   | Time (s) |
|--------|--------|-------------|---------------|---------|----------|
| 3×3    | 1      | 87.1%        | 0.427         | 48,672  | 2309.8   |
| 3×3    | 2      | 79.8%        | 0.681         | 12,168  | 593.5    |
| 3×3    | 3      | 51.3%        | 1.746         | 5,832   | 266.8    |
| 5×5    | 1      | 86.4%        | 0.438         | 115,200 | 1989.1   |
| 5×5    | 2      | 85.0%        | 0.487         | 28,800  | 522.1    |
| 5×5    | 3      | 68.8%        | 1.092         | 12,800  | 222.0    |
| **7×7**| **1**  | **89.3%**    | **0.382**     | 189,728 | 1689.1   |
| 7×7    | 2      | 87.3%        | 0.458         | 47,432  | 420.0    |
| 7×7    | 3      | 73.1%        | 0.880         | 25,088  | 219.1    |

**Best configuration: 7×7 kernel, Stride 1 — 89.3% accuracy, KL = 0.382**

---

## Key Findings

**1. Stride matters more than kernel size**
Changing stride from 1 to 3 drops accuracy from ~89% to 51% — regardless of kernel size. Kernel size alone barely changes results within the same stride.

**2. Larger kernels slightly help**
7×7 consistently outperforms 3×3 and 5×5 at every stride level, but the gain over 5×5 is small (~1%) while compute cost grows ~4x.

**3. Complexity ≠ Training Time**
7×7, Stride=1 has the highest FLOPs but doesn't take the longest to train. Larger kernels produce smaller feature maps, which means fewer inputs to the Dense layer — so the backward pass is actually faster despite higher conv complexity.

**4. Best efficiency trade-off: 7×7, Stride=2**
87.3% accuracy, KL = 0.458, trained in 420 seconds — less than 25% of the cost of 7×7, Stride=1.

---

## What We Used

- Python
- NumPy (only library — no deep learning frameworks)
- MNIST dataset

---
