const http = require("node:http");
const path = require("node:path");
const fs = require("node:fs/promises");

// This server only exposes the finished browser files inside dist.
const publicDirectory = path.resolve(__dirname, "dist");
const requestedPort = Number.parseInt(process.env.PORT || "9123", 10);
const port = Number.isInteger(requestedPort) && requestedPort > 0 && requestedPort < 65536
  ? requestedPort
  : 9123;
const host = process.env.HOST || "127.0.0.1";

const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon"
};

function sendText(response, statusCode, message) {
  // Small errors are returned as plain text; there is no API in this Type-A app.
  response.writeHead(statusCode, {
    "Content-Type": "text/plain; charset=utf-8",
    "Cache-Control": "no-store"
  });
  response.end(message);
}

const server = http.createServer(async (request, response) => {
  // The site is read-only, so only normal page/file requests are accepted.
  if (request.method !== "GET" && request.method !== "HEAD") {
    sendText(response, 405, "Method not allowed");
    return;
  }

  try {
    const requestUrl = new URL(request.url, `http://${request.headers.host || "localhost"}`);
    const decodedPath = decodeURIComponent(requestUrl.pathname);
    const relativePath = decodedPath === "/" ? "index.html" : decodedPath.replace(/^\/+/, "");
    let filePath = path.resolve(publicDirectory, relativePath);

    // Stop ../ path tricks from escaping the public dist folder.
    if (filePath !== publicDirectory && !filePath.startsWith(`${publicDirectory}${path.sep}`)) {
      sendText(response, 403, "Forbidden");
      return;
    }

    const stats = await fs.stat(filePath);
    if (stats.isDirectory()) filePath = path.join(filePath, "index.html");

    // Read and return the requested HTML, CSS, JavaScript, or image file.
    const body = await fs.readFile(filePath);
    const contentType = contentTypes[path.extname(filePath).toLowerCase()] || "application/octet-stream";
    response.writeHead(200, {
      "Content-Type": contentType,
      "Content-Length": body.length,
      "Cache-Control": "no-cache"
    });
    response.end(request.method === "HEAD" ? undefined : body);
  } catch (error) {
    if (error.code === "ENOENT" || error.code === "ENOTDIR") {
      sendText(response, 404, "Page not found");
      return;
    }

    console.error(error);
    sendText(response, 500, "Something went wrong");
  }
});

server.listen(port, host, () => {
  console.log(`Customer Team Finder is running at http://localhost:${port}`);
  console.log("Press Ctrl+C to stop it.");
});

function shutDown() {
  server.close(() => process.exit(0));
}

process.on("SIGINT", shutDown);
process.on("SIGTERM", shutDown);
