const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const api=require('../dist/svm.js');
const m=JSON.parse(fs.readFileSync(path.join(__dirname,'../dist/model.json'),'utf8'));
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-8,a+' != '+b);
for(const row of m.training_examples){close(api.decisionScore(row.fasting_glucose_mg_dl,row.bmi,m),row.score);assert.equal(api.predict(row.fasting_glucose_mg_dl,row.bmi,m).numeric_label,row.numeric_label);}
const bmi=29,z2=(bmi-m.means[1])/m.scales[1];
for(const target of [-1.5,-0.5,0.5,1.5]){const glucose=m.means[0]+m.scales[0]*(target-m.bias-m.weights[1]*z2)/m.weights[0],r=api.predict(glucose,bmi,m);close(r.score,target);assert.equal(r.numeric_label,target>0?1:-1);assert.equal(r.inside_margin,Math.abs(target)<1);const p=api.projection(r.z,m);close(m.weights[0]*p[0]+m.weights[1]*p[1]+m.bias,0);close((r.z[0]-p[0])*m.weights[1]-(r.z[1]-p[1])*m.weights[0],0);}
assert.equal(api.zoneFromScore(0,m),m.zones.higher);
for(const invalid of [[0,25],[NaN,25],[100,Infinity]])assert.throws(()=>api.predict(...invalid,m));
assert.throws(()=>api.predict(100,25,{...m,scales:[0,1]}));
console.log('Browser linear SVM: Python score parity, binary decisions, margins, and perpendicular geometry passed.');
