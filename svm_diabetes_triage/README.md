# Linear SVM decision lab

This update follows **svm15.pdf**, Andrew W. Moore's Support Vector Machines lecture: binary linear classification, maximum margins, support vectors, the quadratic dual and soft-margin slack. The existing **40-row synthetic glucose/BMI problem** supplies numerical data; the lecture's sketches do not specify a table of coordinates.

The full algorithm is in **svm_model.py**, separate from train_model.py and the webpage. It uses a **linear kernel** and sequential minimal optimization (SMO) to solve the dual problem with finite C. Python's standard library is sufficient.

## Run

```powershell
cd D:\machine_learning_2026\svm_diabetes_triage
python train_model.py
python start_app.py
```

Open the printed address (normally http://127.0.0.1:8001). Keep the terminal open; Ctrl+C stops it. The page fetches dist/model.json, so use the server instead of opening index.html directly.

## The linear decision rule

```text
zj = (xj - training_mean_j) / training_scale_j
score = w1*z1 + w2*z2 + b
score < 0  -> lower_reference, class -1
score >= 0 -> higher_reference, class +1
```

The PDF initially writes w dot x - b, then uses w dot x + b for its planes. This code consistently uses **+b**; the sign convention just reverses the stored bias.

The boundary has score 0. Margin planes have scores -1 and +1; the margin width is 2/||w|| in standardized feature space. Inputs inside the margin are still classified by score sign. The score is not a probability, and the margin is not a third class or a diagnosis. Labels describe synthetic reference groups only.

Training minimizes **0.5 ||w||² + C sum(slack)** with y_i*(w dot z_i+b) >= 1-slack_i. Its dual has 0 <= alpha_i <= C and sum(alpha_i*y_i)=0. Actual support vectors have alpha_i > 1e-8; these are not an arbitrary count of nearest points.

## Change training parameters

```powershell
python train_model.py --C 1 --tolerance 0.00001 --max-passes 30 --max-iterations 20000 --random-seed 7
```

| Option | Default | What changing it does |
| --- | ---: | --- |
| --C | 10.0 | Penalty for margin violations. Larger values penalize violations more; smaller values favor wider margins and allow more violations. Must be finite and positive. |
| --tolerance | 0.0001 | KKT optimality tolerance. Smaller values request a closer optimization fit and may need more iterations. |
| --max-passes | 20 | Stop after this many complete scans with no alpha updates. |
| --max-iterations | 10000 | Maximum number of full training scans; prevents an unbounded solve. |
| --random-seed | 42 | Reproducible candidate ordering when selecting multiplier pairs. |
| --dataset | data/diabetes_triage.csv | Input CSV with the same feature columns and labels. |
| --output | dist/model.json | Output model; the page loads dist/model.json. |

After changing parameters, retrain and refresh the page. It displays the settings, convergence status, support count and margin width. Check **Converged: True** in training output. If the iteration cap stops the solver early, the exported model is approximate and the page reports that status. Restore defaults with **python train_model.py**.

Change glucose or BMI in the form and select **Classify point** to update the input diamond and the score calculation without retraining. **Download graph (SVG)** saves the current classification plot.

To change the training problem, edit data/diabetes_triage.csv or use --dataset. Columns are fasting_glucose_mg_dl, bmi, label. Measurements must be finite and positive. Labels must be lower_reference or higher_reference, with both classes present and at least four records. Feature means and scales are recalculated from this data.

Python use:

```python
from svm_model import SimpleLinearSVM, load_dataset
model = SimpleLinearSVM(C=10.0, tolerance=0.0001)
model.fit(load_dataset("data/diabetes_triage.csv"))
print(model.predict(112, 29))
```

## Read the graph

- Axes are standardized glucose and BMI with equal geometry; this is the space where the SVM maximizes its margin.
- Blue circles represent class -1; red squares represent class +1.
- Rings identify points with nonzero learned dual coefficients, including slack support vectors if present.
- The solid boundary has score 0. Dashed margin lines have scores -1 and +1.
- The gold diamond is your input. Its dashed segment ends at the perpendicular projection onto the boundary.
- The explanation shows scaling, weighted score, sign and signed distance score/||w||.

The default solve classifies all 40 training examples correctly and identifies three support vectors. This is training accuracy, not held-out validation.

## Files

| File | Purpose |
| --- | --- |
| svm_model.py | Scaling, linear dual SMO training, supports, binary prediction and export |
| train_model.py | Training CLI and model generation |
| data/diabetes_triage.csv | Existing synthetic two-class dataset |
| dist/svm.js | Separate browser classifier and projection math |
| dist/app.js | Form, score explanation, support table and SVG drawing |
| dist/index.html / styles.css | Page structure and presentation |
| dist/model.json | Generated weights, bias, scaling and dual coefficients |
| start_app.py | Local server; --port changes its initial port |

## Verify

```powershell
python -m unittest discover -s tests -v
node tests/test_browser_model.js
```

Checks cover a known maximum-margin solution, dual feasibility, KKT residuals, binary predictions inside the margin, C's effect, support coefficients, invalid inputs, exports, Python/browser score parity and perpendicular projection geometry.

## Saved graphs

Open [the decision graph](graphs/linear_svm_decision.png) or its [SVG version](graphs/linear_svm_decision.svg) directly. These show the default model and input. Regenerate them after retraining:

```powershell
python -m pip install -r requirements-plot.txt
python plot_model.py
python plot_model.py --glucose 150 --bmi 33 --output graphs/another_decision.png
```

The optional plotting script reads dist/model.json and saves PNG and SVG versions. Use --model to plot an alternate export. Matplotlib is needed only for saved plots; the training scripts and browser SVG graphs require no additional Python packages.
