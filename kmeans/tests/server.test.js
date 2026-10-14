const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const net = require("node:net");
const { spawn } = require("node:child_process");
const { parseCSV } = require("../dist/data.js");

test("local server serves the new app, worker and three real CSVs", async t => {
  const probe = net.createServer();
  await new Promise(resolve => probe.listen(0,"127.0.0.1",resolve));
  const port = probe.address().port;
  await new Promise(resolve => probe.close(resolve));
  const child = spawn(process.execPath,["server.js"],{
    cwd:path.join(__dirname,".."),env:{...process.env,PORT:String(port),HOST:"127.0.0.1"},windowsHide:true
  });
  t.after(()=>child.kill());
  await new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(Error("Server did not start")),5000);
    child.stdout.on("data",chunk=>{if(String(chunk).includes("is running")){clearTimeout(timer);resolve();}});
    child.on("error",error=>{clearTimeout(timer);reject(error);});
    child.on("exit",code=>{clearTimeout(timer);reject(Error("Server exited: "+code));});
  });
  const base="http://127.0.0.1:"+port;
  const page=await fetch(base);assert.equal(page.status,200);assert.match(await page.text(),/Let the data find its centers/);
  for(const [file,count] of [["dataset1.csv",72],["dataset2.csv",90],["dataset3.csv",84]]){
    const response=await fetch(base+"/data/"+file);
    assert.equal(response.status,200);assert.match(response.headers.get("content-type"),/text\/csv/);
    assert.equal(parseCSV(await response.text()).records.length,count);
  }
  for(const file of ["kmeans.js","data.js","train-worker.js"]){assert.equal((await fetch(base+"/"+file)).status,200);}
  assert.equal((await fetch(base+"/missing.csv")).status,404);
  assert.equal((await fetch(base+"/%2e%2e%2fserver.js")).status,403);
});
