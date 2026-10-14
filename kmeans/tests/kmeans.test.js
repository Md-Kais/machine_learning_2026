const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { fit, predict, squaredDistance } = require("../dist/kmeans.js");
const { parseCSV, escapeCSV } = require("../dist/data.js");
const close = (a, b, tolerance = 1e-8) => assert.ok(Math.abs(a - b) <= tolerance, a + " != " + b);
const sortCenters = centers => centers.map(c => c.slice()).sort((a, b) => a[0] - b[0]);

function checkModel(points, model) {
  assert.equal(model.counts.reduce((a, b) => a + b, 0), points.length);
  assert.equal(model.centers.length, model.options.k);
  model.labels.forEach((label, i) => assert.equal(predict(points[i], model).cluster, label));
  let total = 0;
  model.centers.forEach((center, cluster) => {
    const members = model.transformed.filter((point, i) => model.labels[i] === cluster);
    if (model.converged) {
      assert.ok(members.length > 0);
      center.forEach((value, j) => close(value, members.reduce((sum, p) => sum + p[j], 0) / members.length));
    }
    members.forEach(point => total += squaredDistance(point, center));
  });
  close(model.inertia, total);
  close(model.inertia, Math.min(...model.runs.map(run => run.inertia)));
}

test("learns known means from rows rather than predefined centers", () => {
  const points = [[0,0], [0,2], [10,10], [10,12]];
  const model = fit(points, { k: 2, standardize: false });
  assert.deepEqual(sortCenters(model.rawCenters), [[0,1], [10,11]]);
  assert.equal(model.converged, true); close(model.inertia, 4);
  checkModel(points, model);
  const shifted = points.map(point => [point[0] + 20, point[1] - 5]);
  assert.deepEqual(sortCenters(fit(shifted, { k:2, standardize:false }).rawCenters), [[20,-4],[30,6]]);
});

test("K=1 learns the entire dataset mean; K=distinct can isolate rows", () => {
  const points = [[1,2], [3,4], [9,12]];
  const one = fit(points, { k:1 });
  close(one.rawCenters[0][0], 13/3); close(one.rawCenters[0][1], 6);
  checkModel(points, one);
  const three = fit(points, { k:3 }); close(three.inertia, 0); checkModel(points, three);
});

test("all three CSVs learn with different user-selected cluster counts", () => {
  for (const file of ["dataset1.csv", "dataset2.csv", "dataset3.csv"]) {
    const dataset = parseCSV(fs.readFileSync(path.join(__dirname, "../dist/data", file), "utf8"));
    assert.equal(dataset.features.length, 2);
    const points = dataset.records.map(record => record.values);
    for (const k of [1,2,3,4,6]) {
      const model = fit(points, { k }); assert.ok(model.converged); checkModel(points, model);
      const objective = model.history.filter(frame => frame.inertia !== null).map(frame => frame.inertia);
      for (let i=1; i<objective.length; i++) assert.ok(objective[i] <= objective[i-1] + 1e-8);
    }
  }
});

test("seeded attempts are reproducible and do not mutate input", () => {
  const points = [[1,2],[3,2],[4,5],[9,8],[8,7]], copy = JSON.stringify(points);
  assert.deepEqual(fit(points,{k:2,seed:7}),fit(points,{k:2,seed:7}));
  assert.equal(JSON.stringify(points),copy);
});

test("scaling is learned from the CSV and constant features stay finite", () => {
  const model=fit([[1,5],[3,5],[5,5]],{k:1});
  close(model.scaler.means[0],3); close(model.scaler.means[1],5);
  close(model.scaler.scales[0],Math.sqrt(8/3)); close(model.scaler.scales[1],1);
  assert.ok(Number.isFinite(predict([3,5],model).distance));
});

test("prediction compares all center distances with deterministic ties", () => {
  const model=fit([[0,0],[0,2],[10,0],[10,2]],{k:2,standardize:false});
  const result=predict([1,1],model);
  close(result.distance,1); assert.equal(result.distances.length,2);
  const tie=predict([5,1],model); assert.equal(tie.cluster,0);
});

test("iteration cap is reported instead of claiming convergence", () => {
  const model=fit([[0,0],[1,0],[2,0],[10,0],[12,0]],{k:2,maxIterations:1,nInit:1,standardize:false});
  assert.equal(model.iterations,1); assert.equal(model.converged,false);
  assert.equal(model.history.at(-1).phase,"final"); checkModel([[0,0],[1,0],[2,0],[10,0],[12,0]],model);
});

test("duplicate rows and invalid learning settings are handled", () => {
  const points=[[0,0],[0,0],[10,10],[10,10]];
  checkModel(points,fit(points,{k:2}));
  for (const settings of [{k:0},{k:1.5},{k:3},{k:2,seed:-1},{k:2,nInit:0},{k:2,tolerance:NaN},{k:2,maxIterations:0}]) assert.throws(()=>fit(points,settings));
  assert.throws(()=>fit([[0,Infinity]],{k:1})); assert.throws(()=>fit([],{k:1}));
  assert.throws(()=>predict([NaN,1],fit(points,{k:2})));
});

test("CSV supports BOM, quoted ids, scientific notation and optional ids", () => {
  const data=parseCSV('\uFEFFid,x,y\r\n"row, one",1.5,2e1\r\n"row ""two""",-2,3\r\n');
  assert.equal(data.records[0].id,"row, one"); assert.deepEqual(data.records[0].values,[1.5,20]);
  assert.equal(data.records[1].id,'row "two"');
  assert.deepEqual(parseCSV('a,b\n1,2\n').records,[{id:"1",values:[1,2]}]);
  assert.equal(escapeCSV('a,"b"'),'"a,""b"""');
});

test("malformed CSV cannot silently replace missing numbers with zero", () => {
  for (const text of ['x,y\n1,\n','x,y\nNaN,2\n','x,y\nInfinity,2\n','id,x,y\na,1,2\na,3,4\n','x,x\n1,2\n','x,y,z\n1,2,3\n','x,y\n"1,2\n','x,y\n1,2,3\n']) assert.throws(()=>parseCSV(text));
});
