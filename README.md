# Decision Trees

## Introduction

Decision Trees are versatile machine learning algorithms capable of handling both classification and regression tasks. They are particularly powerful, as they can fit complex datasets while maintaining interpretability.

Additionally, Decision Trees form the building blocks of ensemble methods such as Random Forests and gradient-boosting algorithms like XGBoost, which rank among the most effective machine learning techniques available today.

At their core, tree-based methods partition the feature space into a series of rectangular regions, within which a simple model, often a constant value, is fitted. These methods, while conceptually straightforward, are highly effective. A common implementation is CART (Classification and Regression Trees), which simplifies the terminology and unifies the approach for both classification and regression problems.

Let’s illustrate the concept using a regression problem with a continuous response variable $Y$ and two input variables, $X_1$ and $X_2$, each constrained to the unit interval. To simplify, we focus on recursive binary partitioning:

1. First, the feature space is split into two regions. Within each region, the response is modeled by the mean of $Y$. The split variable and the split-point are chosen to maximize the fit.
2. Subsequently, one or both regions are further split, and this process continues until a predefined stopping criterion is met.

The resulting regression model predicts $Y$ as a constant $c_m$ within each region $R_m$, expressed mathematically as:


$$\hat{f}(X) = \sum_{m=1}^{M} c_m \cdot I\{(X_1, X_2) \in R_m\}$$


where $I$ is an indicator function that equals 1 if the condition is true and 0 otherwise.

One of the key advantages of recursive binary trees lies in their interpretability, making them a popular choice for exploratory data analysis and real-world applications.

## Regression Trees

Assume the dataset consists of $p$ inputs and a response for each of $N$ observations, represented as $(x_i, y_i)$ for $i = 1, 2, \dots, N$, where $x_i = (x_{i1}, x_{i2}, \dots, x_{ip})$. The goal of a regression tree is to predict the response $y_i$ by recursively partitioning the input space and fitting a constant value within each partition.

The algorithm partitions the input space into $M$ non-overlapping regions $R_1, R_2, \dots, R_M$ and models the response in each region as a constant $c_m$. The prediction for an observation $x$ is given by:

$$\hat{y} = \sum_{m=1}^M c_m \cdot \mathbb{I}(x \in R_m),$$

where $\mathbb{I}(\cdot)$ is the indicator function, which is $1$ if $x \in R_m$ and $0$ otherwise.

To construct the tree:
1. **Splitting Criterion**: At each step, the algorithm selects a splitting variable $x_j$ and a split point $s$ that minimizes the total sum of squared residuals (SSR) within the resulting regions:

   $$\text{SSR} = \sum_{m=1}^M \sum_{x_i \in R_m} (y_i - c_m)^2.$$
   
2. **Optimal Constant**: The constant $c_m$ for each region $R_m$ is the mean of the response values in that region:

   $$c_m = \frac{1}{|R_m|} \sum_{x_i \in R_m} y_i.$$

The recursive splitting process continues until a stopping criterion is met, such as a minimum number of observations in a region or a maximum tree depth.

Regression trees are particularly effective for capturing non-linear relationships in the data but may require pruning or ensemble techniques like Random Forests to mitigate overfitting.

 

## Classification Trees

Assume the dataset consists of $p$ inputs and a categorical response for each of $N$ observations, represented as $(x_i, y_i)$ for $i = 1, 2, \dots, N$, where $x_i = (x_{i1}, x_{i2}, \dots, x_{ip})$ and $y_i \in \{1, 2, \dots, K\}$. The goal of a classification tree is to predict the category of $y_i$ by recursively partitioning the input space and assigning a class label to each partition.

The algorithm partitions the input space into $M$ regions $R_1, R_2, \dots, R_M$ and assigns a class label $\hat{y}_m$ to each region. The prediction for an observation $x$ is given by:

$$\hat{y} = \hat{y}_m \quad \text{where } x \in R_m.$$

### Tree Construction
1. **Splitting Criterion**: At each step, the algorithm selects a splitting variable $x_j$ and a split point $s$ that minimizes the impurity measure in the resulting regions. Common impurity measures include:
   - **Gini Index**:
     $$\text{Gini}(R) = \sum_{k=1}^K p_k (1 - p_k),$$
     where $p_k$ is the proportion of observations in region $R$ belonging to class $k$.
   - **Entropy**:
     $$\text{Entropy}(R) = -\sum_{k=1}^K p_k \log(p_k).$$
   - **Classification Error**:
     $$\text{Error}(R) = 1 - \max_k(p_k).$$

2. **Assigning Class Labels**: Each region $R_m$ is assigned the majority class label based on the highest proportion of observations in that region:
   $$\hat{y}_m = \arg\max_k p_k.$$

### Advantages
- Handles categorical responses effectively.
- Captures non-linear relationships between features and responses.

### Limitations
- Prone to overfitting without pruning or regularization.
- Less stable as small changes in the data can lead to different trees.

Classification trees are often improved by ensemble methods like Random Forests and Gradient Boosting, which enhance their accuracy and robustness.


## Application

Decision Trees are powerful tools for both regression and classification tasks. In regression, they partition the feature space into regions and assign constant values, effectively capturing non-linear relationships. For classification, they split the feature space using criteria like Gini impurity or entropy, making them intuitive and interpretable. To see practical examples of using Decision Trees for both tasks, refer to this notebook: [**Decision Trees Example Notebook**](https://github.com/apexcds/decision_tree/blob/main/Decision%20Trees.ipynb).
