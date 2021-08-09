About
=====

ML as a Data Production Service
--------------------------------

Today’s society increasingly resorts to algorithms for decision-making and resource allocation. For example, judges in the US make legal decisions aided by predictions from supervised machine learning algorithms. Supervised learning is also used by governments to detect potential criminals and terrorists, and by banks and insurance companies to screen potential customers. Tech companies like Facebook, Microsoft, and Netflix allocate digital content by reinforcement learning and bandit algorithms. Retailers and e-commerce platforms engage in algorithmic pricing. Similar algorithms are encroaching on high-stakes settings, such as in education, healthcare, and the military

All of the above, seemingly diverse examples share a common trait: an algorithm makes decisions based only on its observable input variables. Thus conditional on the observable variables, algorithmic treatment decisions are (quasi-)randomly assigned. This property makes algorithm-based treatment decisions **into instrumental variables (IVs) we can use for measuring the causal effect of the final treatment assignment**. The algorithm-based instrument may produce regression-discontinuity-style local variation (e.g. machine judges), stratified randomization (e.g. certain bandit and reinforcement leaning algorithms), or some combination of the two.

The sources of causal-effect identification can be summarized by a suitable modification of the Propensity Score, dubbed the Approximate Propensity Score (APS) by :cite:`Narita2021`. For each covariate value :math:`x`, the Approximate Propensity Score is the average probability of a treatment recommendation in a shrinking neighborhood around :math:`x`.

Please refer to the section :doc:`method` for a formal introduction to the causal estimation framework.

MLisNE Pipeline
----------------

The IVaps package is an implementation of the treatment effect estimation method described above and in :doc:`method`. This package provides functions for Approximate Propensity Score (APS) estimation and treatment effect estimation that is ML framework-agnostic.

The package serves the two key steps for causal effect estimation in this framework:

- **APS estimation**: This package provides functions for efficient APS estimation with both ONNX and custom user function support.
- **IV estimation**: This package provides functions for IV estimation and counterfactual value estimation wrappers around estimators from the `linearmodels package <https://bashtage.github.io/linearmodels/>`_.

In addition, the helpers module provides additional tools for ONNX assistance, such as a framework-agnostic ONNX conversion function and a function for executing ONNX runtime sessions.
