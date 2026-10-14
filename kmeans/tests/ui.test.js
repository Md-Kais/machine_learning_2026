/* Integration checks for CSV -> worker training -> page output, without a browser dependency. */
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const base = path.join(__dirname, "../dist");

class Element {
  constructor(tag) {
    this.tag = tag; this.children = []; this.attrs = {}; this.handlers = {}; this.style = {};
    this.value = ""; this.checked = false; this.disabled = false; this.hidden = false;
    this.textContent = ""; this.tBodies = []; this.files = [];
  }
  setAttribute(key,value) { this.attrs[key] = String(value); }
  append(...children) { this.children.push(...children); }
  prepend(...children) { this.children.unshift(...children); }
  replaceChildren(...children) { this.children = children; this.textContent = ""; }
  addEventListener(event,handler) { this.handlers[event] = handler; }
  createTHead() { this.tHead = new Element("thead"); this.append(this.tHead); return this.tHead; }
  createTBody() { const body = new Element("tbody"); this.tBodies.push(body); this.append(body); return body; }
  insertRow() { const row = new Element("tr"); this.append(row); return row; }
  insertCell() { const td = new Element("td"); this.append(td); return td; }
  click() { if (this.handlers.click) this.handlers.click({}); }
  cloneNode() {
    const node = new Element(this.tag); node.attrs = {...this.attrs}; node.textContent=this.textContent;
    node.children=this.children.map(child=>typeof child==="string"?child:child.cloneNode()); return node;
  }
}
const text = node => typeof node === "string" ? node : node.textContent + node.children.map(text).join(" ");
const flush = () => new Promise(resolve => setImmediate(resolve));
const escape = value => String(value).replaceAll("&","&amp;").replaceAll('"',"&quot;").replaceAll("<","&lt;").replaceAll(">","&gt;");
function serialize(node) {
  if (typeof node === "string") return escape(node);
  return "<"+node.tag+Object.entries(node.attrs).map(([k,v])=>' '+k+'="'+escape(v)+'"').join("")+">"+escape(node.textContent)+node.children.map(serialize).join("")+"</"+node.tag+">";
}

async function page() {
  const elements = {}, downloads = [], workers = [];
  const html = fs.readFileSync(path.join(base,"index.html"),"utf8");
  for (const match of html.matchAll(/<([a-z-]+)[^>]*id="([^"]+)"[^>]*>/g)) {
    const el = new Element(match[1]); elements[match[2]] = el;
    const value = match[0].match(/value="([^"]*)"/); if(value)el.value=value[1];
    el.checked = /\bchecked\b/.test(match[0]);
    el.disabled = /\bdisabled\b/.test(match[0]);
    if (match[1] === "table") { el.createTHead(); el.createTBody(); }
  }
  elements.dataset.value = "dataset1.csv";
  const document = {
    getElementById: id => { assert.ok(elements[id], "Missing page element: " + id); return elements[id]; },
    createElement: tag => new Element(tag), createElementNS: (ns,tag) => new Element(tag), createTextNode: value => String(value)
  };
  class Worker {
    constructor(filename) {
      this.stopped=false; workers.push(this);
      const self={postMessage:message=>queueMicrotask(()=>{if(!this.stopped)this.onmessage({data:message});})};
      const scope=vm.createContext({self});
      scope.importScripts=file=>vm.runInContext(fs.readFileSync(path.join(base,file),"utf8"),scope);
      vm.runInContext(fs.readFileSync(path.join(base,filename),"utf8"),scope);
      this.scope=scope;
    }
    postMessage(message) { queueMicrotask(()=>{if(!this.stopped)this.scope.self.onmessage({data:message});}); }
    terminate() { this.stopped=true; }
  }
  const context = vm.createContext({
    document, Worker, console, Blob, setTimeout:()=>0, setInterval:()=>1, clearInterval:()=>{},
    URL: {createObjectURL:blob=>{downloads.push(blob);return "blob:test";},revokeObjectURL:()=>{}},
    XMLSerializer: class {serializeToString(node){return serialize(node);}}, navigator:{},location:{protocol:"http:"},
    fetch:async filename=>({ok:true,text:async()=>fs.readFileSync(path.join(base,filename),"utf8")})
  });
  for (const filename of ["kmeans.js","data.js","app.js"]) vm.runInContext(fs.readFileSync(path.join(base,filename),"utf8"),context);
  await flush();
  const submit=id=>elements[id].handlers.submit({preventDefault(){}});
  return {elements,downloads,workers,submit};
}

test("page loads a CSV, trains in its actual worker, and explains a prediction",async()=>{
  const p=await page(),e=p.elements;
  assert.match(e["data-summary"].textContent,/72 rows/); assert.equal(e.train.disabled,false);
  assert.equal(e.predict.disabled,true);
  p.submit("train-form"); await flush();
  assert.equal(e["cluster-count"].textContent,"3"); assert.equal(e.predict.disabled,false);
  assert.match(text(e["prediction-result"]),/Cluster/);
  assert.match(e["mean-explanation"].textContent,/mean =/);
  assert.equal(e.distances.tBodies[0].children.length,3);
  assert.ok(e["cluster-graph"].children.length>650);
  assert.equal(e.error.textContent,"");
  e["query-x"].value="80";e["query-y"].value="50";p.submit("predict-form");
  assert.match(e["selected-id"].textContent,/New input/);
  assert.match(e["distance-math"].textContent,/sqrt/);
  e.step.value="0"; e.step.handlers.input();
  assert.match(e["step-description"].textContent,/Initialization/);
  e.next.click(); assert.match(e["step-description"].textContent,/Assign rows/);
  e.next.click(); assert.match(e["step-description"].textContent,/Update means/);
  e.final.click();assert.match(e["step-description"].textContent,/Final result/);
});

test("changing K actually retrains, and changing CSV invalidates the previous model",async()=>{
  const p=await page(),e=p.elements;
  e.k.value="4";p.submit("train-form");await flush();
  assert.equal(e.centers.tBodies[0].children.length,4);
  assert.equal(e.distances.tBodies[0].children.length,4);
  e.dataset.value="dataset3.csv";await e.dataset.handlers.change();
  assert.equal(e.predict.disabled,true);assert.match(e["data-summary"].textContent,/84 rows/);
  assert.equal(e["feature-x-label"].textContent,"temperature c");
  e.k.value="2";p.submit("train-form");await flush();
  assert.equal(e["cluster-count"].textContent,"2");
  assert.equal(e.assignments.tBodies[0].children.length,84);
});

test("upload, invalid K, cancellation and result exports use the same trained model",async()=>{
  const p=await page(),e=p.elements;
  e.upload.files=[{name:"own.csv",size:45,text:async()=>"id,x,y\na,0,0\nb,0,2\nc,10,10\nd,10,12\n"}];
  await e.upload.handlers.change();assert.match(e["data-summary"].textContent,/4 rows/);
  e.k.value="5";p.submit("train-form");assert.match(e.error.textContent,/distinct points/);
  e.k.value="2";p.submit("train-form");e.cancel.click();await flush();
  assert.match(e.status.textContent,/cancelled/);assert.equal(e.predict.disabled,true);
  p.submit("train-form");await flush();assert.equal(e.centers.tBodies[0].children.length,2);
  e["download-labels"].click();e["download-centers"].click();e["download-svg"].click();
  const [labels,centers,svg]=await Promise.all(p.downloads.map(blob=>blob.text()));
  assert.equal(labels.trim().split(/\r?\n/).length,5);
  assert.match(labels,/distance_to_center/);assert.match(centers,/assigned_rows/);
  assert.match(svg,/<svg/);assert.match(svg,/xmlns=/);
  assert.equal(e.error.textContent,"");
});
