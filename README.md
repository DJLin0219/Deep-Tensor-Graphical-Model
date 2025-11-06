# DETECT  
**Deep Tensor Graphical Model for Sparse Graph Recovery**

This repository contains the preliminary implementation of **DETECT**,  
a deep tensor graphical model designed for sparse graph estimation  
and high-dimensional precision matrix learning.  

The current release focuses on the algorithmic core and code structure.  
A full version including theoretical background, benchmark experiments,  
and tensor-based extensions will be released in a future update.

---

## 🔍 Overview  

**DETECT** introduces a deep unrolled optimization framework that learns  
the sparse structure of conditional dependencies among variables.  
It generalizes classical graphical model estimation to tensor-valued settings  
by embedding the optimization process into a differentiable neural architecture.

Key features include:  
- **Tensor-aware modeling:** preserves high-order structural dependencies.  
- **Adaptive optimization:** learns hyperparameters dynamically through backpropagation.  
- **Unrolled Alternating Minimization:** interpretable iteration-by-iteration updates.  
- **Differentiable regularization:** integrates soft-thresholding within neural layers.  
- **GPU-accelerated training:** implemented in PyTorch for scalability.  

---

## 🧩 Code Structure  
