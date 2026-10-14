/* Binary linear SVM inference and geometry, independent of the UI. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;root.SimpleSvm=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
 'use strict';
 function validateModel(m){if(!m||m.model_type!=='linear_svm'||!['weights','means','scales'].every(k=>Array.isArray(m[k])&&m[k].length===2&&m[k].every(Number.isFinite))||m.scales.some(s=>s<=0)||!Number.isFinite(m.bias))throw Error('Missing or invalid linear SVM model.');}
 function standardize(glucose,bmi,m){validateModel(m);if(![glucose,bmi].every(v=>Number.isFinite(v)&&v>0))throw Error('Glucose and BMI must be finite numbers greater than zero.');return [(glucose-m.means[0])/m.scales[0],(bmi-m.means[1])/m.scales[1]];}
 function decisionScore(glucose,bmi,m){const z=standardize(glucose,bmi,m);return m.weights[0]*z[0]+m.weights[1]*z[1]+m.bias;}
 function zoneFromScore(score,m){if(!Number.isFinite(score))throw Error('Score must be finite.');return score>=0?m.zones.higher:m.zones.lower;}
 function predict(glucose,bmi,m){const score=decisionScore(glucose,bmi,m),norm=Math.hypot(...m.weights);return {score,zone:zoneFromScore(score,m),label:score>=0?'higher_reference':'lower_reference',numeric_label:score>=0?1:-1,inside_margin:Math.abs(score)<1,signed_distance:norm?score/norm:null,z:standardize(glucose,bmi,m)};}
 function projection(z,m){validateModel(m);const norm2=m.weights[0]**2+m.weights[1]**2;if(!norm2)return null;const score=m.weights[0]*z[0]+m.weights[1]*z[1]+m.bias;return z.map((v,i)=>v-score*m.weights[i]/norm2);}
 return {standardize,decisionScore,zoneFromScore,predict,projection};
});
