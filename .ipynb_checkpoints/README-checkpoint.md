# ft_linear_regression
An introduction to machine learning.

The program that predicts the price of a car by using a linear function train with a gradient descent algorithm.


---

Statistical method used to model the relationship between a *dependent* variable and one or more *independent* variable.
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

> [!question]
Why we add $\frac{1}{2}$ coefficient to the equation
> 


```desmos-graph
left=-1; right=6
bottom=-1; top=6;
---
j=(x-3)^2 + 2|label:j(a)
x=3|y>0|y<2|label:j(a)|dotted
(3,2)|label:'a'|cross|black
```
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
[[Linear regression - Matrix notation]]

![[linear_reg_equations.png]]

---
## References
- [Linear regression - Wikipedia](https://en.wikipedia.org/wiki/Linear_regression)
- [What is Linear Regression?- Spiceworks - Spiceworks](https://www.spiceworks.com/tech/artificial-intelligence/articles/what-is-linear-regression/#:~:text=Linear%20regression%20is%20a%20statistical,Last%20Updated%3A%20April%203%2C%202023)
- [Machine Learnia - LA RÉGRESSION LINÉAIRE (partie 1/2) - ML#3 - YouTube](https://www.youtube.com/watch?v=wg7-roETbbM&list=PLO_fdPEVlfKqUF5BPKjGSh7aV9aBshrpY&index=3)

## References
- [Project Jupyter | Home](https://jupyter.org/)