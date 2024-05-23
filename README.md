# numerical-tube-manifolds
In this repository you will find the code that I developed for my bachelor thesis' project.
The aim of the work was to find the tube manifolds of periodic orbits of the *circular restricted three body problem* (CRTBP). 
I wanted to use as little math as possible, therefore I rely heavily on numerical methods for my calculations.
Some knowledge of linear algebra, dynamical systems and basic numerical methods (Newton's method) should be enough to follow the example provided and understand the main concepts.

I'm aware that there are many other methods to find the tube manifolds, more elegant and efficient but also more mathematically complex. I wanted to build a methodology that would allow a STEM student of undergraduate level (as was I when I first approached this topic) to calculate these complex trajectories.

You can find the final report of my thesis [here](https://www.researchgate.net/publication/380514048_Numerical_calculation_of_manifolds_of_periodic_orbits_of_the_restricted_three_body_problem), which you can use as reference. It is divided into three parts: the necessary theory, the other two explain in detail how to calculate the orbits of the *planar* and *spatial* problem.

In the repository you can find:
1. The code I've developed to do the calculations.
2. A notebook with a simple example on the tube manifolds of a vertical Lyapunov orbit

## Aknowledgments
My thanks go to professor Christos Efthymiopoulos, who has patientelly thought me the complex math of tube manifolds and Hamiltonian dynamics, as well as some tricks in numerical computation.
