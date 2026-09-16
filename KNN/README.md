# BMI Neighbor — a tiny KNN project

This project teaches **K-Nearest Neighbors (KNN)** with a BMI example. You enter
an adult's height and weight, the app calculates BMI, and five nearby examples
from a CSV dataset vote for one category:

- Underweight
- Normal
- Overweight
- Obese

It uses only normal Python, HTML, CSS, and JavaScript. There is no ML library to
hide the important ideas.

> **Important:** This is a learning project, not a medical diagnosis. Adult BMI
> is a screening measurement. Children and teenagers need age- and sex-specific
> growth charts, so this predictor is for adults age 20 and older.

## Run it

1. Open a terminal in this project folder.
2. Train and export the model:

   ```powershell
   python train_model.py
   ```

3. Start the BMI website:

   ```powershell
   python start_app.py
   ```

4. Open <http://localhost:8000> in a browser.

Keep that terminal open while using the website. Press `Ctrl+C` in the terminal
when you want to stop the server. `start_app.py` finds the `dist` folder by
itself, so the command also works when it is launched using the full script path
from another folder.

## Offline use

The app contains its model, code, styles, and fonts locally; it makes no API
calls. Start it with `python start_app.py` even when the computer has no internet
connection. After one successful visit on localhost or a secure host, `sw.js`
also saves the required public files so the browser can reopen the installed page
offline. Opening `index.html` directly is not supported because browsers block
`fetch("model.json")` on many `file://` pages.

## Deploy to Vercel

`vercel.json` already selects `dist` as the public output and makes the
service worker refresh safely. After signing in to the Vercel CLI, run:

```powershell
cd KNN
vercel
```

Choose the defaults when first linking the project. Use `vercel --prod` for a
production release. There is no build command, inference server, environment
variable, or paid dependency.

## The whole idea in plain words

Imagine you have a new dot and many labeled dots on a number line. KNN finds the
five labeled dots closest to the new dot. Each nearby dot votes. The label with
the most votes becomes the answer. The webpage also places those people on a
normal x/y graph so you can see their real height and weight.

In this project:

1. Python reads height and weight examples from the CSV file.
2. It calculates a BMI for each example.
3. `fit()` remembers those examples. That is KNN's simple training step.
4. The model is saved as `dist/model.json` for the browser.
5. JavaScript repeats the same distance-and-vote steps when a user submits the form.

The distance is just:

```text
distance = absolute value of (user BMI - example BMI)
```

## How to read the neighbor graph

- The **x-axis** goes left to right and shows height in centimetres.
- The **y-axis** goes bottom to top and shows weight in kilograms.
- Every small colored dot is one row from `data/bmi_data.csv`.
- The lime diamond is the example person or the person using the form.
- The five large ringed and numbered points are the neighbors that vote.
- Each neighbor is labeled with its height, weight, and BMI directly on the graph.
- Dashed lines make it easy to follow the person to those five dots.

The model chooses neighbors by **BMI distance**, not by the shortest-looking line
on the height-and-weight graph. This is useful: two people with different heights
and weights can still have almost the same BMI.

## File-by-file summary

### `data/bmi_data.csv`

This is the dataset. Each row contains a height, weight, and category. The rows
include examples close to important BMI boundaries so KNN can learn where one
category changes into another. CSV is used because it looks like a simple table
and is easy to edit in a text editor or spreadsheet.

### `knn_model.py`

This is the actual machine-learning code. It calculates BMI, loads the CSV, finds
the five closest examples, counts their votes, and returns a category. It is
written from scratch so you can read every KNN step instead of relying on a large
library. It also exports the learned examples to JSON for the website.

### `train_model.py`

This is the file you run. It temporarily keeps some data aside for a small test,
prints the accuracy, trains again with all rows, and creates `dist/model.json`.
Keeping the training job in its own short file makes the order of the steps easy
to follow.

### `start_app.py`

This starts the local website at `http://127.0.0.1:8000`. It always finds the
project's `dist` folder, even if the command is run from a different folder, and
prints a beginner-friendly message if port 8000 is already busy. It uses only
Python's built-in web server, so nothing extra must be installed.

### `dist/model.json`

This generated file is the trained model that the browser reads. KNN does not
learn a complicated formula; it remembers labeled examples. That is why the JSON
contains `k` plus each example's height, weight, BMI, and category. The extra
height and weight values let the webpage draw the x/y graph. Do not edit it by hand—run
`python train_model.py` instead.

### `dist/index.html`

This is the structure and wording of the user interface. It contains the height
and weight form, result area, BMI scale, x/y neighbor graph, three-step lesson,
measurement help, and the medical disclaimer. Semantic HTML and labels make it
usable with keyboards and screen readers.

### `dist/styles.css`

This controls the visual design. It creates the responsive two-card layout,
color-coded result states, graph-paper background, large readable controls, and
small result animations. It also styles the chart axes, category legend, ringed and
labeled neighbors, dashed connections, and responsive phone layout. A reduced-motion rule
respects users who turn animations off in their device settings.

### `dist/knn.js`

This is the browser's small, testable KNN brain. It contains the BMI formula,
distance calculation, neighbor selection, and voting. Keeping the maths separate
from the buttons and animations makes it much easier to compare with
`knn_model.py` and to test without opening a browser.

### `dist/app.js`

This connects the HTML controls to `knn.js` and the exported model. It validates
the inputs, asks the KNN brain for an answer, fills the result card, and starts
the animations. It draws the SVG x/y graph from the real exported dataset rows
and redraws it whenever the inputs or screen size change. The five closest points
get larger rings, rank numbers, and measurement labels inside the graph. In browsers that support WebMCP, it also registers one
`predict_bmi_category` tool that performs the same visible action.

### `tests/test_knn.py`

These are small automatic checks for the BMI formula, all four categories, bad
input, and JSON export. They also confirm that exported rows include the height
and weight coordinates required by the graph. They use Python's built-in
`unittest`, so there is nothing extra to install.

### `tests/test_browser_model.js`

This checks the KNN code that actually runs in the webpage. It verifies the BMI
formula, all four categories, the five returned neighbors, their chart
coordinates, and error handling. It uses Node.js's built-in tools, so it needs no
JavaScript package installation.

### `dist/sw.js` and `dist/manifest.webmanifest`

These make the hosted lesson installable and cache its local files for later
offline use. They do not change the KNN calculation.

### `vercel.json`

This tells Vercel to publish the `dist` folder and always re-check `sw.js` for
updates. It does not affect the KNN algorithm.

### `.gitignore`

This keeps temporary Python cache files out of version control. Those files are
created automatically and are not part of the project.

## Try the Python model directly

Open Python in this folder and run:

```python
from knn_model import SimpleKNN, load_dataset

examples = load_dataset("data/bmi_data.csv")
model = SimpleKNN(k=5)
model.fit(examples)
print(model.predict(height_cm=170, weight_kg=65))
```

## Change the project

- Change `k=5` to another positive odd number, such as `3` or `7`, and compare results.
- Add sensible rows to `data/bmi_data.csv`, then run `python train_model.py` again.
- Change the category colors inside `dist/styles.css`.
- Change the friendly result messages inside `dist/app.js`.

The usual adult BMI boundaries represented by this teaching dataset are below
18.5, 18.5–24.9, 25–29.9, and 30 or higher. Real medical decisions need more
information than BMI alone.

## Health information references

- [CDC adult BMI categories](https://www.cdc.gov/bmi/adult-calculator/bmi-categories.html)
- [CDC child and teen BMI calculator](https://www.cdc.gov/bmi/child-teen-calculator/index.html)
