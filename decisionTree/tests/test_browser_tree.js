const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const api=require('../dist/tree.js');
const model=JSON.parse(fs.readFileSync(path.join(__dirname,'../dist/model.json'),'utf8'));
const question={Outlook:'Sunny',Temp:'Mild',Humidity:'High',Wind:'Strong'};
const answer=api.predict(question,model);
assert.equal(answer.label,'No');
assert.deepEqual(answer.path.map(s=>[s.feature,s.value]),[['Outlook','Sunny'],['Humidity','High']]);
for(const row of model.training_examples)assert.equal(api.predict(row,model).label,row.Play);
for(const Outlook of model.categories.Outlook)for(const Temp of model.categories.Temp)for(const Humidity of model.categories.Humidity)for(const Wind of model.categories.Wind){const result=api.predict({Outlook,Temp,Humidity,Wind},model);const expected=Outlook==='Overcast'?'Yes':Outlook==='Sunny'?(Humidity==='Normal'?'Yes':'No'):(Wind==='Weak'?'Yes':'No');assert.equal(result.label,expected);}
assert.throws(()=>api.predict({...question,Outlook:'Snow'},model));
assert.throws(()=>api.predict(question,{}));
console.log('Browser ID3: all 36 weather combinations and lecture paths passed.');
