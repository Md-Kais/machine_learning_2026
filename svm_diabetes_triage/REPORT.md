# Linear SVM report

The current implementation follows the binary linear-classifier, margin and dual-optimization material in svm15.pdf by Andrew W. Moore. The existing 40-row synthetic dataset provides glucose and BMI coordinates. svm_model.py contains the algorithm; train_model.py, browser inference and drawing remain separate.

Features are standardized using training means and scales. A linear soft-margin SVM solves the dual with sequential minimal optimization. Class is determined by the sign of w dot z + b; the margin does not create an additional class. Support vectors are identified from nonzero learned dual multipliers.

With default C=10 and tolerance=0.0001, training converges with 100% training accuracy and three support vectors. The page shows the boundary, both margin planes, class regions, support vectors, input and its perpendicular projection. It also shows the exact score calculation, signed distance, solver status and support coefficients. Downloadable SVG graphs preserve the current decision.

This is a synthetic classification lesson. Its reference classes are not diagnoses, and training accuracy is not held-out validation. See README.md for settings, commands and checks.
