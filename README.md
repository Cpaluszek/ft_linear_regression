# ft_linear_regression
An introduction to machine learning.

The program that predicts the price of a car by using a linear function train with a gradient descent algorithm.


---

Linear Regression is a statistical method used to model the relationship between a *dependent* variable and one or more *independent* variable.
The goal is to find the best-fitting linear relationship that can be used to make predictions.

![25-4.png (720×624)|400](https://images.spiceworks.com/wp-content/uploads/2022/04/07040339/25-4.png)

**Mathematical equation:**
$Y=mX+b$

**Machine learning:**
$y(x) = p0 + p1.x$


## Multiple linear regression
$y(x) = p0 + p1x1 + p2x2 + ... + p(n)x(n)$

To determine the line best fits the data, the model evaluates different weight combinations.
The model uses a cost function to optimize the weights (pi). The cost function of linear regression is the root mean squared error or mean squared error (MSE).

$MSE = \frac{1}{N}\sum_{i=1}^{n} (y_i-(mx_i+b)^2)$


## Precision
-> [Regression and performance metrics — Accuracy, precision, RMSE and what not! | by Ullas Kakanadan | Analytics Vidhya | Medium](https://medium.com/analytics-vidhya/regression-and-performance-metrics-accuracy-precision-rmse-and-what-not-223348cfcafe)

-> [Regression Metrics for Machine Learning - MachineLearningMastery.com](https://machinelearningmastery.com/regression-metrics-for-machine-learning/)

- R-Squared
- Adjusted R-Squared
- MSE
- RMSE

---
## Example
### Dataset
`(x, y), m = 6, n = 1`

### Model
$f(x) = ax + b$

Start with random `a` and `b` parameters.

### Cost Function
**Mean squared error - Erreur quadratique moyenne:**
$J(a, b) = \frac{1}{2m}\sum(f(x^i) - y^i)^2$

### Minimization algorithm
**Gradient Descent:**
Minimize the cost function derivative
$\frac{\partial J}{\partial a} = 0$

[[Gradient Descent]]

`alpha` - *learning rate*

**Partial derivatives:**
$\frac{\partial J}{\partial a} = \frac{1}{m}\sum(x(ax + b - y))$
$\frac{\partial J}{\partial b} = \frac{1}{m}\sum(ax + b - y)$

$a_{i+1}+=a_i-alpha\frac{\partial J(a_i)}{\partial a}$
$b_{i+1}+=b_i-alpha\frac{\partial J(b_i)}{\partial b}$

## Matrix notation
### Use case
- Matrices allows to express the entire set of equations and calculations in a *compact form*
- Represents multiples features
- *Computationally efficient*

### Example Matrix notation

**Model:**
$f(x) = ax + b$

$F = X . \theta$

$$\begin{bmatrix}
f(x^1) \\
f(x^2) \\
... \\
f(x^m)\end{bmatrix}=\begin{bmatrix}
x^1 & 1\\
x^2 & 1 \\
... & ... \\
x^m & 1\end{bmatrix} \\
\begin{bmatrix}a \\
b\end{bmatrix}$$

Variable count: 1
X dimension = `m x (n + 1)`
θ dimension = `(n + 1) x 1`

**Cost function**

$$
J(\theta) = \frac{1}{2m}\sum(X.\theta-Y)^2
$$

**Gradient:**
$$
\frac{\partial{J(\theta)}}{\partial{\theta}} = \frac{1}{m} X^T(X\theta - Y)
$$

**Gradient descent:**
$$
\theta = \theta - \alpha \frac{\partial{J}}{\partial{\theta}}
$$

---
## References
- [Linear regression - Wikipedia](https://en.wikipedia.org/wiki/Linear_regression)
- [What is Linear Regression?- Spiceworks - Spiceworks](https://www.spiceworks.com/tech/artificial-intelligence/articles/what-is-linear-regression/#:~:text=Linear%20regression%20is%20a%20statistical,Last%20Updated%3A%20April%203%2C%202023)
- [Machine Learnia - LA RÉGRESSION LINÉAIRE (partie 1/2) - ML#3 - YouTube](https://www.youtube.com/watch?v=wg7-roETbbM&list=PLO_fdPEVlfKqUF5BPKjGSh7aV9aBshrpY&index=3)

### Tools
- [Project Jupyter | Home](https://jupyter.org/)
