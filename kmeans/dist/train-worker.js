/* Keep training responsive even when users request many clusters. */
importScripts("kmeans.js");
self.onmessage = event => {
  try { self.postMessage({ model: KMeans.fit(event.data.points, event.data.options) }); }
  catch (error) { self.postMessage({ error: error.message }); }
};
