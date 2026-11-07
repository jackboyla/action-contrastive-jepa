from pathlib import Path
import json,numpy as np
root=Path('experiments/rebuttal/results/rebuttal-scene'); result={}
for method in ['sigreg','masked','masked_action_nce']:
 rows=[json.loads((root/f'{method}_{s}.json').read_text()) for s in [3072,1,2]]
 result[method]={'rank_mean':float(np.mean([r['participation_ratio'] for r in rows])),
 'probes':{k:float(np.mean([np.mean(r['probes'][k]['r2']) for r in rows])) for k in rows[0]['probes']},
 'rollout_nmse':{h:float(np.mean([r['rollouts'][h]['normalized_mse'] for r in rows])) for h in ['1','3','5','10']}}
 reference=np.load(root/'sigreg_3072.npz')
 for s in [3072,1,2]:
  with np.load(root/f'{method}_{s}.npz') as a:
   for k in ['frame_ids','split','window_starts','window_episodes']:
    assert np.array_equal(a[k],reference[k]),(method,s,k)
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
