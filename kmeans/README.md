# Customer Team Finder

A simple, interactive K-means customer segmentation lesson built with HTML, CSS, and JavaScript.

## Run on localhost

You need Node.js 18 or newer. No packages need to be installed.

1. Open a terminal in this folder.
2. Run:

   ```powershell
   npm start
   ```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.
4. Press `Ctrl+C` in the terminal when you want to stop the server.

To use a different port in PowerShell:

```powershell
$env:PORT=8080
npm start
```

## Offline use

The HTML, styles, sample data, and K-means algorithm are all local. You can
double-click `dist/index.html`, or run `npm start`, without an internet
connection. On localhost or a secure host, `sw.js` also caches the whole app
after the first visit so an installed copy can reopen offline.

## Deploy to Vercel

`vercel.json` already publishes `dist` and tells browsers to re-check the
service worker for updates. After signing in to the Vercel CLI, run:

```powershell
cd kmeans
vercel
```

Choose the defaults when first linking the project. Use `vercel --prod` for a
production release. The app needs no build command, API, database, or environment
variables.
