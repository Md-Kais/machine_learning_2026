const CACHE_NAME = "dataset-kmeans-v2";
const FILES = ["./", "./index.html", "./styles.css", "./app.js", "./kmeans.js", "./data.js", "./train-worker.js", "./manifest.webmanifest", "./data/dataset1.csv", "./data/dataset2.csv", "./data/dataset3.csv"];
self.addEventListener("install", event => {
  event.waitUntil(caches.open(CACHE_NAME).then(cache => cache.addAll(FILES)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", event => {
  event.waitUntil(caches.keys().then(names => Promise.all(names.filter(name => name === "customer-team-finder-offline-v1" || (name.startsWith("dataset-kmeans-") && name !== CACHE_NAME)).map(name => caches.delete(name)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", event => {
  if (event.request.method !== "GET" || new URL(event.request.url).origin !== self.location.origin) return;
  event.respondWith(fetch(event.request).then(response => {
    if (response.ok) {
      const clone = response.clone();
      event.waitUntil(caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone)));
    }
    return response;
  }).catch(async () => (await caches.match(event.request)) || Response.error()));
});
