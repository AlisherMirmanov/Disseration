## Acceleration in optimization: perspective from geometry and classical physics
# Abstract
This project serves as a literature review on physical and geometrical interpretations of
a popular modern optimization technique, Nesterov Gradient Descent (NGD), developed by
Yurii Nesterov in [10], which has been shown to achieve a quadratic convergence rate. Af-
ter defining the setting of the investigation, we state and compare NGD with conventional
Gradient Descent (GD). Further, in order to better understand its accelerated convergence
rate, the continuous time limit of the algorithm is derived, following the work proposed in
[3] and [14]. The system is analyzed under the calculus of variation framework, showing the
exponential convergence rate of O( 1
eβ) for the ODE. Additionally, as in [15], by focusing on
a specific β - smooth quadratic function, certain conditions are imposed on the ODE, for it
to truly optimize action. For practical application, a general form of discretization of rate-
matching Nesterov ODE is presented, with an arbitrary convergence rate of O( 1
kp) in normed
spaces, a special case of which is in fact NGD. To offer insights into Nesterov optimization in
a non-Euclidean setting, the extension of all these results in a manifolds setting is considered,
first developed in [1] and later by [5], showing that the ODE achieves exponential convergence
rate for geodesically convex functions, and extending the discretization of the ODE. Lastly,
the penalty function approximation problem and Batch Normalization problem are used for
illustration, which could be solved using the methods explored in the paper.
