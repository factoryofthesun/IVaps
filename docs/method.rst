MLisNE Framework Overview
=========================
[INSERT OVERVIEW HERE]


Examples
--------

The MLisNE method can be applied to a broad swath of algorithms which generate treatment recommendations. Below are a few examples of popular algorithms for which our framework applies.

Supervised Learning
~~~~~~~~~~~~~~~~~~~~

Millions of times each year, judges make bail-or-release decisions that hinge on a prediction of what a defendant would do if released. Many judges now use proprietary algorithms (like COMPAS criminal risk score) to make such predictions and use the predictions to support bail-or-release decisions. Kleinberg et al. (2017) also developed another prediction algorithm.

These algorithms fit into our framework as a simple special case. Using our notation, assume that a criminal risk algorithm recommends bailing (:math:`Z_i=1`) and releasing (:math:`Z_i=0`) to each defendent *i*. The algorithm uses defendant *i*'s observable characteristics :math:`X_i`, includinng criminal history and demographics. The algorithm first translates :math:`X_i` into a continuous risk score :math:`r(X_i)`, where :math:`r:\mathbb{R}^p \rightarrow \mathbb{R}` is a function estimated by supervised learning based on past data and assumed to be fixed.
