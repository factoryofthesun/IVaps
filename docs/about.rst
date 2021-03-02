About
=====

ML as a Data Production Service
--------------------------------

Today’s society increasingly resorts to machine learning (“AI”) and other algorithms for decision-making and resource allocation. For example, judges make legal judgements using predictions from supervised machine learning (descriptive regression). Supervised learning is also used by governments to detect potential criminals and terrorists, and financial companies (such as banks and insurance companies) to screen potential customers. Tech companies like Facebook, Microsoft, and Netflix allocate digital content by reinforcement learning and bandit algorithms. Uber and other ride sharing services adjust prices using their surge pricing algorithms to take into account local demand and supply information. Retailers and e-commerce platforms like Amazon engage in algorithmic pricing. Similar algorithms are invading into more and more high-stakes treatment assignment, such as education, health, and military.

All of the above, seemingly diverse examples share a common trait: An algorithm makes decisions based only on observable input variables the data-generating algorithm uses. Conditional on the observable variables, therefore, algorithmic treatment decisions are (quasi-)randomly assigned. This property makes algorithm-based treatment decisions **an instrumental variable we can use for measuring the causal effect of the final treatment assignment**. The algorithm-based instrument may produce regression-discontinuity-style local variation (e.g. machine judges), stratified randomization (e.g. several bandit and reinforcement leaning algorithms), or mixes of the two.

Please refer to the section :doc:`method` for a formal introduction to the causal estimation framework.

MLisNE Pipeline
----------------

The IVaps package is an implementation of the treatment effect estimation method described above. This package provides functions for APS estimation and treatment effect estimation that is ML framework-agnostic.

The package serves the two key steps for causal effect estimation in this framework:

- **APS estimation**: This package provides functions for efficient APS estimation with both ONNX and custom user function support.
- **IV estimation**: This package provides functions for IV estimation and counterfactual value estimation wrappers that use estimators from the `linearmodels package <https://bashtage.github.io/linearmodels/>`_.

In addition, the helpers module provides additional tools for ONNX assistance, such as a framework agnostic ONNX conversion function and a function for executing ONNX runtime sessions.
