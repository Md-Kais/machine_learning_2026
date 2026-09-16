# Behind the Curtain — HMM Lab

This is a small interactive explanation of the hidden two-coin experiment from
Rabiner's HMM tutorial. A learner enters the tosses they heard, changes how the
coins or dealer behave, and sees the most likely hidden coin at every step.

For the project purpose, complete input/output explanation, non-technical
walkthrough, technical usage, worked example, and troubleshooting guide, read
[`REPORT.md`](REPORT.md).

## Where the important pieces live

- `dist/hmm.py` is the detailed Python teaching version of Viterbi decoding and
  forward-backward, with no third-party packages.
- `dist/hmm.js` is its browser-ready twin. It runs the same calculations using
  only local JavaScript, so the page never downloads Pyodide or calls a server.
- `dist/app.js` reads the controls, calls the local HMM module, and draws the
  result. It intentionally contains no mathematical implementation.
- `dist/index.html` contains the page structure.
- `dist/styles.css` contains the responsive visual design.
- `dist/sw.js` caches the app after a hosted visit for offline reopening.
- `tests/test_hmm.py` checks Python against exhaustive path enumeration, while
  `tests/test_browser_hmm.js` checks the browser engine against Python fixtures.

## Run it offline

The app has no remote fonts, libraries, APIs, or model downloads. You can
double-click `dist/index.html` and use it directly. For the same secure-origin
behavior used by hosting—including the installable offline cache—serve it locally:

```powershell
cd dist
python -m http.server 4173
```

Then open `http://127.0.0.1:4173/`. No internet connection is needed.

## Deploy to Vercel

`vercel.json` already selects `dist` as the public output and prevents the
service worker from being stuck in a long browser cache. After signing in to the
Vercel CLI, run this from the HMM folder:

```powershell
vercel
```

Choose the defaults when first linking the project. Future production releases
can use `vercel --prod`. No build command, Python server, or environment
variable is required.

## Run the correctness checks

```powershell
python -m unittest discover -s tests -v
node tests/test_browser_hmm.js
```

## Default model

The prompt gives the two emission probabilities but does not specify the dealer's
transition probabilities. The app therefore uses clearly labeled, editable
defaults: an 80% chance of keeping either coin and a 50/50 starting chance.
