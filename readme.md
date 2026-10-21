# Predicting Cheeger Constants from Graph Spectra

This project investigates how well spectral data can predict the Cheeger constant of a graph. It compares predictions based on the full spectrum with predictions using only the spectral gap.

## Graphs and target values

The dataset consists of 21,635 connected, 6-regular bipartite graphs with 168 vertices, divided into two parts of 84 vertices each.

For a 6-regular graph, we use the normalized Cheeger constant

$$
h(G)=\min_{0<|S|\leq |V|/2}\frac{|\partial S|}{6|S|},
$$

where $\partial S$ is the set of edges joining $S$ to its complement.

Computing the Cheeger constant exactly is computationally difficult in general. We therefore use estimates obtained by a randomized local search. Starting from random vertex subsets, the algorithm searches for improvements by changing the membership of one or two vertices at a time.

Each estimate comes with a witness subset and is an upper bound on the true Cheeger constant. The models are trained and evaluated against these estimates, rather than independently verified exact values.

## Spectral features

We compute the eigenvalues of the normalized Laplacian

$$
L=I-\frac{1}{6}A,
$$

where $A$ is the adjacency matrix. We write the eigenvalues in increasing order:

$$
0=\lambda_1<\lambda_2\leq\cdots\leq\lambda_{168}=2.
$$

The spectral gap is $\lambda_2$. Classical spectral bounds connect it to the Cheeger constant, motivating the use of spectral data for prediction.

We compare two feature sets:

- **Spectrum:** the 83 eigenvalues $\lambda_2,\ldots,\lambda_{84}$.
- **Spectral gap only:** the single eigenvalue $\lambda_2$.

For bipartite graphs, the normalized Laplacian spectrum is symmetric about 1:

$$
\lambda_i+\lambda_{169-i}=2.
$$

Consequently, the selected 83 eigenvalues, together with the known endpoints, determine the full spectrum. The remaining eigenvalues would provide redundant information.

## Models and training

We compare three regression models:

- Linear regression (LR).
- Random forest regression (RF).
- A multilayer perceptron neural network (MLP).

Each model is trained with both feature sets. All experiments use the same reproducible 80/20 train/test split.

Linear regression and MLP use a pipeline with `StandardScaler`. Scaling is learned from training data only. Random forests use the unscaled features.

For RF and MLP, automated hyperparameter tuning uses `GridSearchCV` with five shuffled cross-validation folds within the training set. Candidate settings are compared using mean absolute error (MAE).

The searches vary:

- **RF:** number of trees, maximum tree depth, and minimum samples per leaf.
- **MLP:** hidden-layer sizes, regularization strength, and maximum training iterations.

After selecting the best settings, each search refits the winning model on the entire training set.

## Results

Performance is measured against the held-out Cheeger estimates. Lower MAE and RMSE are better; higher R² is better.

| Model | Features | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Linear regression | Spectrum | 0.012105 | 0.016539 | 0.777164 |
| Linear regression | Spectral gap only | 0.020571 | 0.025457 | 0.472044 |
| Random forest | Spectrum | 0.006377 | 0.010011 | 0.918361 |
| Random forest | Spectral gap only | 0.009164 | 0.015091 | 0.814485 |
| MLP | Spectrum | 0.014073 | 0.018115 | 0.732658 |
| MLP | Spectral gap only | 0.015377 | 0.019724 | 0.683073 |

Already for linear regression, using the spectrum reduces MAE by approximately 41% compared with using the spectral gap alone. This indicates that the other eigenvalues contain useful predictive information beyond the gap.

The full-spectrum random forest achieves the best performance, with an MAE of approximately 0.00638 and R² of 0.918. Its MAE is approximately 30% lower than that of the gap-only random forest.

We also inspect the linear regression coefficients. In the gap-only model, the fitted slope is approximately 0.4823, with an intercept of 0.0531. An increase of 0.01 in the spectral gap therefore corresponds to an increase of approximately 0.00482 in the predicted Cheeger estimate.

In the spectrum model, the largest absolute standardized coefficients belong to $\lambda_2$ and $\lambda_3$, with opposite signs:

| Eigenvalue | Standardized coefficient | Raw coefficient |
|---|---:|---:|
| $\lambda_2$ | 0.046073 | 0.917839 |
| $\lambda_3$ | −0.020288 | −0.383860 |
| $\lambda_8$ | 0.007580 | 0.296135 |
| $\lambda_6$ | 0.007166 | 0.240565 |
| $\lambda_7$ | −0.006868 | −0.237599 |

Standardized coefficients express the predicted change associated with a one-training-standard-deviation increase in an eigenvalue, holding the other features fixed. Raw coefficients express the same relationship in the original eigenvalue units.

The opposite signs of the two leading coefficients suggest that the fitted model uses both the spectral gap and its relationship to the next eigenvalue. Several other nearby eigenvalues also have coefficients with opposite signs, consistent with the model using contrasts in spectral shape.

These are conditional relationships within the fitted model. In particular, the negative coefficient of $\lambda_3$ does not imply that $\lambda_3$ alone is negatively correlated with the Cheeger estimate. Since eigenvalues are correlated, coefficients can compensate for one another and may be unstable. Their magnitudes should therefore not be interpreted as standalone measures of feature importance or as evidence of an exact mathematical formula.

## Scope and limitations

The results demonstrate prediction of locally computed Cheeger estimates within this graph dataset. They do not establish an exact formula for the Cheeger constant or guarantee performance on other graph families.

The test set was inspected during development, so the reported scores should be viewed as exploratory benchmark results.

## AI assistance

ChatGPT was used to discuss machine learning concepts, assist with writing and debugging code, interpret model results, and draft this README. The experiments were run locally by the author, who made the final decisions about the project and its implementation.