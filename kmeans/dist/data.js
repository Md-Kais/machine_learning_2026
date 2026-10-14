/* CSV loading is separate from the clustering algorithm and interface. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.ClusterData = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  function parseCSV(text) {
    if (typeof text !== "string") throw Error("CSV content must be text.");
    text = text.replace(/^\uFEFF/, "");
    const rows = []; let row = [], field = "", quoted = false, closed = false;
    function cell() { row.push(field.trim()); field = ""; closed = false; }
    function line() { cell(); if (row.some(value => value !== "")) rows.push(row); row = []; }
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      if (quoted) {
        if (char === '"' && text[i + 1] === '"') { field += '"'; i++; }
        else if (char === '"') { quoted = false; closed = true; }
        else field += char;
      } else if (char === '"') {
        if (field.trim() || closed) throw Error("Invalid quote in CSV field.");
        field = ""; quoted = true;
      } else if (char === ",") cell();
      else if (char === "\n" || char === "\r") { line(); if (char === "\r" && text[i + 1] === "\n") i++; }
      else { if (closed && char.trim()) throw Error("Unexpected text after a quoted CSV field."); field += char; }
    }
    if (quoted) throw Error("CSV contains an unclosed quoted field.");
    if (field || row.length) line();
    if (rows.length < 2) throw Error("CSV needs a header and at least one data row.");
    const headers = rows.shift();
    const hasID = headers[0].toLowerCase() === "id";
    const features = hasID ? headers.slice(1) : headers;
    if (features.length !== 2 || headers.some(h => !h) || new Set(headers).size !== headers.length) {
      throw Error("CSV needs exactly two feature columns, optionally preceded by an id column.");
    }
    if (rows.length > 1000) throw Error("Use at most 1000 rows in this interactive demo.");
    const records = rows.map((values, i) => {
      if (values.length !== headers.length) throw Error("CSV row " + (i + 2) + " has the wrong number of columns.");
      const numeric = hasID ? values.slice(1) : values;
      if (numeric.some(v => !v || !Number.isFinite(Number(v)))) throw Error("CSV row " + (i + 2) + " needs two finite numeric values.");
      if (hasID && !values[0]) throw Error("CSV row " + (i + 2) + " needs an id.");
      return { id: hasID ? values[0] : String(i + 1), values: numeric.map(Number) };
    });
    if (new Set(records.map(r => r.id)).size !== records.length) throw Error("Row ids must be unique.");
    return { features, records };
  }
  function escapeCSV(value) {
    const text = String(value);
    return /[",\r\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  }
  return { parseCSV, escapeCSV };
});
