"""Exact declared visual-only state-baseline versus RGB-recipe comparison.

No model or archived reference is changed. The temporary comparison view only
aligns one explicitly verified nonphysical marker-alpha scalar.
"""
import copy,json,hashlib
from pathlib import Path
import numpy as np

class AppearanceAllowlist:
    def __init__(self,reference,recipe_path,snapshot_fn,geom_name_fn):
        self.reference=reference;self.snapshot_fn=snapshot_fn;self.geom_name_fn=geom_name_fn;self.recipe_path=Path(recipe_path)
        self.recipe=json.loads(self.recipe_path.read_text());rgb=self.recipe['RGB_recipe']
        if any(rgb.get(k)!=v for k,v in {'camera':'front_pixels','width':224,'height':224,'ob_type':'states','visualize_info':False,'offsamples':4}.items()):raise ValueError('Actual completed RGB source recipe differs')
        self.receipt=None
    def __call__(self,u):
        actual=self.snapshot_fn(u);reference=self.reference
        if set(actual)!=set(reference):raise ValueError('Model snapshot field set changed')
        for key in actual:
            if key!='geom_rgba' and actual[key]!=reference[key]:raise ValueError('Nonallowlisted camera/material/light/body field differs:'+key)
        a=np.asarray(actual['geom_rgba']['values'],dtype=np.dtype(actual['geom_rgba']['dtype']));b=np.asarray(reference['geom_rgba']['values'],dtype=np.dtype(reference['geom_rgba']['dtype']))
        if a.shape!=b.shape or a.dtype!=np.float32 or b.dtype!=np.float32:raise ValueError('Original geom RGBA shape/dtype differs')
        ids=[i for i in range(len(a)) if self.geom_name_fn(u._model,i)=='target_object_0']
        if len(ids)!=1:raise ValueError('Exactly one named target_object_0 marker required')
        gid=ids[0]
        if int(u._model.geom_contype[gid])!=0 or int(u._model.geom_conaffinity[gid])!=0:raise ValueError('Allowed target marker is not collision-disabled/nonphysical')
        if a[gid,3]!=np.float32(0.) or b[gid,3]!=np.float32(.2):raise ValueError('Only exact RGB-hidden0 / original-state-markerfloat32.2 alpha allowed')
        difference=np.argwhere(a!=b)
        if difference.tolist()!=[[gid,3]]:raise ValueError('Additional geom RGBA changes forbidden')
        self.receipt={'whole_model_exact_claim':False,'all_other_model_fields_and_components_exact':True,'named_geom':'target_object_0','geom_id':gid,
          'collision_proof':{'contype':0,'conaffinity':0},'actual_alpha':float(a[gid,3]),'reference_alpha':float(b[gid,3]),
          'source_RGB_recipe_sha256':hashlib.sha256(self.recipe_path.read_bytes()).hexdigest(),'source_RGB_recipe':self.recipe['RGB_recipe'],
          'actual_snapshot':actual,'original_reference_snapshot':reference,'changed_indices':difference.tolist(),
          'reason':'intentional RGB training visualizationFalse hides nonphysical goal marker; original state baseline visualizationTrue, state input/physics unchanged',
          'actual_model_modified':False,'archived_reference_modified':False}
        view=copy.deepcopy(actual);adjusted=a.copy();adjusted[gid,3]=b[gid,3]
        payload=str(adjusted.dtype).encode()+str(adjusted.shape).encode()+adjusted.tobytes()
        view['geom_rgba']={'dtype':str(adjusted.dtype),'shape':list(adjusted.shape),'values':adjusted.tolist(),'sha256':hashlib.sha256(payload).hexdigest()}
        if view!=reference:raise ValueError('Allowlist comparison did not leave all other model bytes exact')
        return view
