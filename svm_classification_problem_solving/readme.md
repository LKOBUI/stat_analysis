# SVM Classification Problem Solving

**Repository Link:** [LKOBUI/stat_analysis](https://github.com/LKOBUI/stat_analysis/tree/main)
**Target Directory:** `svm_classification_problem_solving`

### Repository Learning Objectives
* Master the structural foundations of Support Vector Machines (SVM) for separating discrete, multi-class spaces.
* Understand the geometric transition from linear boundary lines to high-dimensional non-linear planes using kernel functions.
* Evaluate the structural balance between widening the margin boundary and limiting misclassification penalties in noisy datasets.

### Detailed Notebook Breakdowns

#### 1. Maximum Margin Linear Classifier (`linear_svm_boundary.ipynb`)
* **Objective:** Construct an optimal linear separating hyperplane that maximizes the geometric distance between two distinct, fully separable classes.
* **Model in Use & Justification:** **Linear Support Vector Classifier (SVC)**. This framework is selected because it directly computes the widest possible margin corridor between groups, focusing exclusively on the most critical edge observations (support vectors) rather than the overall group averages.
* **Mathematical Formulation:**
  The separating boundary is defined by a linear plane equation:
  $$\mathbf{w}^T \mathbf{x} + b = 0$$
  Where $\mathbf{w}$ represents the orientation vector of the plane and $b$ controls the vertical intercept offset. The optimization engine maximizes the margin width, which mathematically scales as:
  $$\text{Margin} = \frac{2}{\|\mathbf{w}\|}$$
  To find the widest margin, the system minimizes the inverse quadratic objective function subject to zero training errors:
  $$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{subject to } y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1$$

#### 2. Soft Margin Optimization & Penalty Tuning (`soft_margin_svc.ipynb`)
* **Objective:** Establish robust classification boundaries in noisy datasets where classes overlap and cannot be perfectly separated by a clean line.
* **Model in Use & Justification:** **Soft Margin Support Vector Classifier (C-SVC)**. This approach is chosen to balance the width of the margin with a penalty for misclassified data points, ensuring the model generalizes well to unseen data.
* **Mathematical Formulation:**
  The model introduces positive slack variables ($\xi_i$) to measure how far an observation falls on the wrong side of the margin boundary. The optimization target shifts to include a cost hyperparameter ($C$):
  $$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i \quad \text{subject to } y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i$$
  Where a large value of $C$ enforces a narrow margin with few training errors, while a small value of $C$ allows a wider margin with more tolerated mistakes to prevent overfitting.

#### 3. Non-Linear Spaces & Dual Kernel Mapping (`kernel_svm_spaces.ipynb`)
* **Objective:** Separate complex, twisting, or concentric data structures that cannot be divided effectively using a straight linear boundary.
* **Model in Use & Justification:** **Kernelized Support Vector Machine (KSVM)**. This model is required because it uses mathematical kernel transformations to project the data into a higher-dimensional space where a flat separating plane can be calculated efficiently.
* **Mathematical Formulation:**
  By converting the optimization problem into its dual form, the model replaces raw coordinate multiplications with a Kernel Function $K(\mathbf{x}_i, \mathbf{x}_j)$, avoiding the high computational cost of transforming coordinates explicitly:
  $$f(\mathbf{x}) = \sum_{i=1}^{n} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$$
  The notebook evaluates two primary kernel functions to shape the non-linear boundaries:
  * Polynomial Kernel Function: $K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + r)^d$
  * Radial Basis Function (RBF / Gaussian Kernel): $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$
  Where $\gamma$ defines how far the influence of a single support vector reaches across the feature space.

### Performance Metrics Captured
* **Support Vector Counts:** Tracks the density and location of boundary points to audit model stability and over-parameterization risks.
* **Geometric Margin Width:** Measures the distance separating the support hyperplanes to evaluate boundary confidence.
* **Classification Boundary Spreading:** Maps training and testing precision-recall curves to ensure non-linear kernels are not chasing localized data noise.
