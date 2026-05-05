# simple_MLP
Assignment for Final Project (Simple MLP)

This notebook implements a Multi-Layer Perceptron (MLP) neural network to predict whether a U.S. county will experience high or low levels of gun violence.

The model is part of a broader machine learning project that aims to build an early warning system using socioeconomic, demographic, and social vulnerability data.

The primary goal is to answer:
- Can we accurately predict whether a county has a high level of gun violence based on socioeconomic and vulnerability indicators?

We used GroupShuffleSplit based on FIPS codes to prevent leakage across counties.
This ensures that Counties in training data do not appear in test data and the model performance reflects generalization to unseen counties

We implemented a feedforward neural network using MLPClassifier from scikit-learn.

Baseline Configuration:
Hidden layers: (50,)
Activation: ReLU
Solver: Adam
Max iterations: 500

