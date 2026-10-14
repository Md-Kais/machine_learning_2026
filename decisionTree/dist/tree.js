/* Categorical ID3 inference, separate from the form and drawing code. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;root.DecisionTree=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
  'use strict';
  function predict(values,model){
    if(!model?.tree||model.model_type!=='id3')throw Error('Missing ID3 model. Run train_model.py.');
    for(const feature of model.features){if(!model.categories[feature].includes(values[feature]))throw Error('Choose a valid '+feature+'.');}
    let node=model.tree;const path=[];
    while(!node.is_leaf){
      const value=values[node.feature], child=node.children[value];
      path.push({node_id:node.id,feature:node.feature,value,gain:node.gain,fallback:!child});
      if(!child)break;
      node=child;
    }
    return {label:node.prediction,leaf_id:node.id,path,fallback:!!path.at(-1)?.fallback};
  }
  return {predict};
});
