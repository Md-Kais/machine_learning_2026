# Behind the Curtain HMM Lab — Project Report

## 1. Problem Title & Purpose

### Title

**Guessing Which Hidden Coin Made a Visible Heads-and-Tails Story**

### Architecture checklist

**Type A: Client-Side Browser ML.** The site does not call an inference server.
`dist/hmm.js` runs Viterbi and forward-backward natively in the browser, and
`dist/app.js` passes it the current controls. The matching `dist/hmm.py` remains
the extra-detailed teaching and Python-test version. There is no exported
`model.json` because the learner builds the HMM from the current sliders.

### Why it matters

Many real processes are hidden while their clues are visible: weather behind
sensor readings, speech sounds behind audio waves, or system health behind logs.
A Hidden Markov Model (HMM) combines clues over time instead of judging each clue
alone. This two-coin game makes that difficult idea safe, small, and visible.

## 2. How the Algorithm Works (10-Year-Old Explanation)

### The story metaphor

A dealer stands behind a curtain holding two coins:

- the **Fair** coin makes heads half the time by default;
- the **Biased** coin makes heads 85% of the time by default.

Before each toss, the dealer may keep the same coin or switch. We hear only `H`
or `T`, never the coin's name. The coin is the **hidden state**, and the heard
result is the **observation**.

Imagine two stepping stones at every time: one labeled Fair and one Biased. A
route across the stones receives points for:

1. how likely its first coin was;
2. how likely each coin was to make the heard result; and
3. how likely the dealer was to keep or switch coins.

The **Viterbi algorithm** finds the strongest complete route without trying every
full route. At each stone it keeps only the strongest path that ends there, plus
a little arrow pointing to the previous winning stone. Paths ending at the same
stone have identical choices ahead, so a weaker one can never catch up.

### Inputs and outputs

| Item | Everyday meaning | Allowed value |
|---|---|---:|
| Observed tosses | The H/T results heard outside the curtain | 1–60 symbols |
| Fair heads | Chance the Fair coin produces H | 5–95% |
| Biased heads | Chance the Biased coin produces H | 5–95% |
| Keep Fair | Chance the dealer keeps Fair after Fair | 5–95% |
| Keep Biased | Chance the dealer keeps Biased after Biased | 5–95% |
| Initial Fair | Chance the first hidden coin is Fair | Fixed at 50% in UI |
| Viterbi output | One strongest start-to-finish coin route | Fair/Biased at each toss |
| Posterior output | Fair/Biased shares at one selected toss | Two percentages totaling 100% |

### The three probability boxes

- **Initial distribution π** is the starting bowl: the chance of beginning with
  either coin.
- **Transition matrix A** is a road-sign table: from the current coin, how likely
  is each next coin?
- **Emission matrix B** is a sound-maker table: if this coin is hidden, how likely
  is the H or T we heard?

A probability is a slice of one whole pie. Each row totals 1 because the next
coin must be Fair or Biased, and a toss must be Heads or Tails.

### Step-by-step magic

1. JavaScript cleans the typed sequence and rejects symbols other than H/T and
   friendly separators.
2. Slider percentages such as 85 become decimals such as 0.85.
3. `make_model()` builds π, A, and B. Missing opposite chances are `1 - value`.
4. Viterbi initialization scores Fair and Biased at the first toss:
   starting chance × emission chance.
5. For every later toss and possible current coin, Viterbi compares the route
   arriving from Fair with the route arriving from Biased.
6. It keeps the stronger incoming score and saves the winning previous-coin arrow.
7. The current emission chance is included.
8. At the final toss, the better of the two ending routes wins.
9. The algorithm follows saved arrows backward, then reverses the list to obtain
   the chronological hidden route.
10. Forward-backward separately passes all route evidence left-to-right and
    right-to-left.
11. Multiplying those two views and normalizing produces Fair/Biased posterior
    bars for each selected toss.
12. The local JavaScript model returns the result object; `app.js` renders the
    decoded badge, timeline, route probability, explanation, and probability bars.

### Key code breakdown

```python
initial_log_score = (
    math.log(initial_probability)
    + math.log(emission_probability)
)
```

For the first stone, starting chance and sound-making chance are multiplied.
Logarithms change multiplication into addition. This keeps a long chain from
shrinking into a number too tiny for the computer to hold accurately.

```python
incoming_log_scores = {
    previous_state: delta[t - 1][previous_state]
    + math.log(model["transitions"][previous_state][current_state])
    for previous_state in STATES
}
```

To stand on today's Fair stone, compare two roads: yesterday's Fair route plus a
Fair→Fair road, and yesterday's Biased route plus a Biased→Fair road. The same
comparison is made for today's Biased stone.

```python
best_previous_state = max(
    STATES,
    key=lambda previous_state: incoming_log_scores[previous_state],
)
```

`max` keeps the stronger road and records which previous stone supplied it. This
saved answer is called a back-pointer.

```python
reversed_path = [final_state]
for time_index in range(len(observations) - 1, 0, -1):
    previous_state = psi[time_index][reversed_path[-1]]
    reversed_path.append(previous_state)
most_likely_path = list(reversed(reversed_path))
```

Start at the winning final stone and follow the chalk arrows backward. Because
that lists the route from finish to start, reverse it before showing the user.

```python
posterior[state] ∝ forward[state] * backward[state]
```

The forward score knows everything heard up to one toss. The backward score
knows everything still to come. Combining both is like reading clues from the
left and right pages before judging the middle page.

### Viterbi path versus posterior bars

Viterbi selects one complete route as a team. Forward-backward adds evidence
from every possible route and asks about one time step. The largest individual
bar can occasionally disagree with the state in the best complete route; that
is a real difference between the two questions, not a bug.

## 3. Simple Web Development Guide

### Folder and file map

```text
HMM/
├── dist/
│   ├── index.html          Toss controls, presets, result badge, and timeline
│   ├── styles.css          Responsive two-coin visual design
│   ├── app.js              Validation, local-model bridge, and result rendering
│   ├── hmm.js              Offline browser HMM algorithms
│   ├── hmm.py              Matching, fully explained Python algorithms
│   ├── sw.js               Saves required files for offline reopening
│   └── manifest.webmanifest Installable-app name, colors, and start page
├── tests/
│   ├── test_hmm.py         Checks results against exhaustive path search
│   └── test_browser_hmm.js Checks browser/Python parity fixtures
├── vercel.json             Publishes dist and safely refreshes the worker
├── README.md               Local running notes
└── REPORT.md               This guide
```

### How it works together

1. `index.html` collects the H/T sequence and four probabilities. Helper text,
   native slider fences, a sequence warning, and four presets reduce guesswork.
2. `index.html` loads the local `hmm.js` engine before `app.js`; there is no
   CDN, API, WebAssembly download, or inference server.
3. On Decode, `app.js` gives one JSON string to `analyseJson()`. The local
   engine builds the HMM and runs both algorithms.
4. `app.js` parses the result and updates the status badge, timeline, route
   summary, path probability, and selected-step bars.
5. On localhost or Vercel, `sw.js` caches the six required public files so a
   previously opened app can reopen with the network switched off.

### Key functions

| Function | Role |
|---|---|
| `parseSequence()` | Accepts friendly H/T formatting and rejects mistakes. |
| `runExperiment()` | Calls the local browser model and handles UI states. |
| `hmm.py: viterbi()` / `hmm.js: viterbi()` | Finds the strongest route in both engines. |
| `forward_backward()` / `forwardBackward()` | Computes per-step posterior shares. |

## 4. Kid-Friendly User Manual

### Quick summary

Type the heads/tails sounds heard from behind the curtain, then press **Decode**.
The page shows its best guess for which hidden coin made each toss.

### Three steps

1. **Type the observations** — enter 1–60 H/T letters or press a ready-made
   sequence such as **Heads streak** or **Alternating**.
2. **Click the button** — wait for **Offline model ready**, then press **Decode**.
3. **Read the badge** — read “Decoded … tosses,” the hidden route, and select any
   toss to inspect its two probability bars.

### Understanding the result

| Badge/color | Plain meaning |
|---|---|
| Teal — Decoded | The local model successfully found and displayed a route. |
| Gold — Decoding | The browser's local model is working. |
| Red — Check/could not decode | The sequence is invalid or calculation failed. |
| Gold Fair state | Viterbi chose the Fair coin at that step. |
| Teal Biased state | Viterbi chose the Biased coin at that step. |
| Fair/Biased bars | Per-step shares using all observations; together they total 100%. |

Repeated heads usually support the biased coin under the defaults; repeated
tails support the fair coin. Higher “keep” settings make the hidden route switch
coins less often.
