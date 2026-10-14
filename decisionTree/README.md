# ID3 Play Tennis

This project follows **15.+Decision+tree+_+ID3.pdf**: the 14-row Tennis Playing table on page 2 and the question on page 3. Training uses categorical **ID3**, base-2 entropy and information gain. Every attribute value gets its own branch. The algorithm is in **decision_tree_model.py**, separate from the training script and webpage.

## Run

From PowerShell:

```powershell
cd D:\machine_learning_2026\decisionTree
python train_model.py
python start_app.py
```

Open the address printed by the server (normally http://127.0.0.1:8002). Keep the terminal open; Ctrl+C stops it. Python's standard library is sufficient. Run the server rather than opening index.html directly, because the browser fetches model.json.

## The exact lecture problem

Inputs: **Outlook=Sunny, Temp=Mild, Humidity=High, Wind=Strong**.

Decision: **Outlook=Sunny -> Humidity=High -> Play=No**.

The complete tree is Outlook -> Sunny/Humidity, Overcast/Yes, Rain/Wind. Sunny days use Humidity (High=No, Normal=Yes); rainy days use Wind (Weak=Yes, Strong=No).

Root entropy is 0.940286 bits. Root information gains are:

| Attribute | Gain (bits) |
| --- | ---: |
| Outlook | 0.246750 |
| Humidity | 0.151836 |
| Wind | 0.048127 |
| Temp | 0.029223 |

The implementation calculates weighted entropy directly from the records. Some arithmetic on the slides has typographical errors; the weighted sum and information gain formulas on pages 18-19 produce the values above.

## Change training parameters

```powershell
python train_model.py --max-depth 1 --min-samples-split 3 --min-gain 0.05
```

| Option | Default | What changing it does |
| --- | --- | --- |
| --max-depth | unlimited | Root depth is 0. Set 0 for one majority leaf, 1 for root-only splits. Limiting depth can change predictions. |
| --min-samples-split | 2 | A node with fewer records stops and uses its majority label. Must be an integer >= 2. |
| --min-gain | 0.0 | Minimum information gain, in bits, required to split. Higher values stop low-gain splits. |
| --dataset | data/play_tennis.csv | Read a different CSV with the same column names and categorical values. |
| --output | dist/model.json | Destination for the trained model. The page reads dist/model.json. |

After retraining, refresh the page. Its graph and displayed parameters come from the newly exported model. To restore the full lecture tree, run **python train_model.py** without options.

Change today's weather through the four dropdowns; this changes the prediction without retraining. The gold path updates automatically. The **Restore PDF question** button resets the original question. **Download graph (SVG)** saves the current tree with its highlighted path.

CSV columns: Outlook, Temp, Humidity, Wind, Play. Accepted values:

- Outlook: Sunny, Overcast, Rain
- Temp: Hot, Mild, Cool
- Humidity: High, Normal
- Wind: Weak, Strong
- Play: Yes, No

Edit data/play_tennis.csv to change training examples, then retrain. A valid category absent from a trained branch uses that node's majority and is explicitly shown as a fallback. Tied majorities choose No; tied gains choose the original feature order. Pure nodes, exhausted attributes, and configured stopping limits end recursion. The older cardiac_triage.csv is retained as a legacy asset and is unused by this ID3 example.

Python use:

```python
from decision_tree_model import ID3DecisionTree, load_dataset
model = ID3DecisionTree(max_depth=None, min_samples_split=2, min_gain=0.0)
model.fit(load_dataset("data/play_tennis.csv"))
print(model.predict({"Outlook": "Sunny", "Temp": "Mild", "Humidity": "High", "Wind": "Strong"}))
```

## Files

| File | Purpose |
| --- | --- |
| decision_tree_model.py | Entropy, gain, categorical training, prediction, export |
| train_model.py | CLI parameters, data loading and model generation |
| data/play_tennis.csv | The lecture's 14 original records |
| dist/tree.js | Separate browser inference algorithm |
| dist/app.js | Form, explanation and highlighted SVG tree |
| dist/index.html / styles.css | Page structure and presentation |
| dist/model.json | Generated trained model; regenerate rather than editing |
| start_app.py | Local static server; --port changes the initial port |

## Verify

```powershell
python -m unittest discover -s tests -v
node tests/test_browser_tree.js
```

Checks cover the PDF's gains, split order, exact question, all training labels, stopping settings, fallback behavior, export and all 36 possible browser weather combinations. The reported 100% is training accuracy on the lecture example, not a generalization estimate.

## Saved graphs

Open [the decision graph](graphs/id3_decision.png) or its [SVG version](graphs/id3_decision.svg) directly. These show the default model and input. Regenerate them after retraining:

```powershell
python -m pip install -r requirements-plot.txt
python plot_model.py
python plot_model.py --outlook Rain --temp Mild --humidity High --wind Weak --output graphs/another_decision.png
```

The optional plotting script reads dist/model.json and saves PNG and SVG versions. Use --model to plot an alternate export. Matplotlib is needed only for saved plots; the training scripts and browser SVG graphs require no additional Python packages.
