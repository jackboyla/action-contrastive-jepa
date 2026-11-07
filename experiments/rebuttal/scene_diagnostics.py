"""Shared-sample Scene checkpoint diagnostics; outputs are local research artifacts."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2


def encode(model, pixels, device, batch=32):
    result = []
    with torch.inference_mode():
        for start in range(0, len(pixels), batch):
            x = torch.from_numpy(pixels[start:start+batch]).permute(0, 3, 1, 2).to(device)
            if not getattr(model, 'preprocess_pixels', False):
                x = x.float()/255
                x = (x-torch.tensor([.485,.456,.406], device=device)[None,:,None,None])/torch.tensor([.229,.224,.225], device=device)[None,:,None,None]
                x = v2.functional.resize(x, [224,224], antialias=True)
            result.append(model.encode({'pixels': x[:,None]})['emb'][:,0].float().cpu().numpy())
    return np.concatenate(result)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--frames',type=int,default=6000)
    p.add_argument('--windows',type=int,default=128)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    rng=np.random.default_rng(20261106)
    source=ROOT/'.stable_worldmodel/ogbench/visual_scene_play.h5'
    with h5py.File(source) as ds:
        offsets=ds['ep_offset'][:];lengths=ds['ep_len'][:]
        episodes=rng.permutation(len(lengths))
        split_eps=np.split(episodes,[int(.6*len(episodes)),int(.8*len(episodes))])
        ids=[]; splits=[]
        for split,eps in enumerate(split_eps):
            count=[int(args.frames*.6),int(args.frames*.2),args.frames-int(args.frames*.8)][split]
            # Uniform episode sampling avoids long trajectories dominating the probe.
            chosen=rng.choice(eps,size=count,replace=True)
            rows=offsets[chosen]+(rng.random(count)*lengths[chosen]).astype(int)
            rows=np.unique(rows)
            ids.extend(rows);splits.extend([split]*len(rows))
        order=np.argsort(ids);ids=np.asarray(ids)[order];splits=np.asarray(splits)[order]
        pixels=ds['pixels'][ids]
        factors={'cube_position':ds['privileged_block_0_pos'][ids],
                 'drawer':ds['privileged_drawer_pos'][ids][:,None],
                 'window':ds['privileged_window_pos'][ids][:,None],
                 'button_0':ds['button_state_0'][ids][:,None],
                 'button_1':ds['button_state_1'][ids][:,None]}
        # MuJoCo qpos quaternion order wxyz; rotation matrices remove q/-q ambiguity.
        q=ds['privileged_block_0_quat'][ids]
        factors['cube_rotation']=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix().reshape(-1,9)
        valid=split_eps[2][lengths[split_eps[2]]>65]
        weps=rng.choice(valid,size=args.windows,replace=True)
        starts=offsets[weps]+(rng.random(args.windows)*(lengths[weps]-65)).astype(int)
        # Match Scene planning: one observed frame, then ten five-action transitions.
        wids=starts[:,None]+5*np.arange(11)[None,:]
        unique,inverse=np.unique(wids,return_inverse=True)
        wpixels=ds['pixels'][unique]
        action_all=ds['action'][:]
        finite=action_all[~np.isnan(action_all).any(1)]
        mean=finite.mean(0,dtype=np.float64);std=finite.std(0,ddof=1,dtype=np.float64)
        if np.any(std==0): raise ValueError('constant action dimension')
        actions=np.stack([action_all[s:s+50] for s in starts])
        actions=((actions-mean)/std).reshape(args.windows,10,-1).astype(np.float32)
    checkpoint=Path(args.checkpoint)
    model=torch.load(checkpoint,map_location=args.device,weights_only=False).eval()
    if not hasattr(model,'encode'): raise TypeError(type(model))
    print('CHECKPOINT',checkpoint,flush=True)
    z=encode(model,pixels,args.device)
    wz=encode(model,wpixels,args.device)[inverse].reshape(args.windows,11,-1)
    train,val,test=[np.flatnonzero(splits==i) for i in range(3)]
    scaler=StandardScaler().fit(z[train]);zs=scaler.transform(z)
    probes={}
    for name,y in factors.items():
        trials=[]
        for alpha in [.01,.1,1.,10.,100.,1000.]:
            reg=Ridge(alpha=alpha).fit(zs[train],y[train])
            score=r2_score(y[val],reg.predict(zs[val]),multioutput='uniform_average')
            trials.append((score,alpha,reg))
        _,alpha,reg=max(trials,key=lambda t:t[0])
        prediction=reg.predict(zs[test])
        probes[name]={'alpha':alpha,'r2':r2_score(y[test],prediction,multioutput='raw_values').tolist()}
        if name.startswith('button'):
            probes[name]['balanced_accuracy']=balanced_accuracy_score(y[test].ravel(),(prediction.ravel()>=.5).astype(int))
    centered=z[test].astype(np.float64)-z[test].mean(0)
    covariance=centered.T@centered/(len(test)-1)
    trace=float(np.trace(covariance))
    rank=trace**2/float(np.square(covariance).sum()) if trace>0 else 0.
    errors=[]; persistence=[]
    with torch.inference_mode():
        emb=torch.from_numpy(wz[:,:1]).to(args.device)
        acts=torch.from_numpy(actions).to(args.device)
        for h in range(1,11):
            aemb=model.action_encoder(acts[:,:h])
            pred=model.predict(emb[:,-3:],aemb[:,-3:])[:,-1:]
            emb=torch.cat([emb,pred],1)
            truth=torch.from_numpy(wz[:,h]).to(args.device)
            errors.append((pred[:,0]-truth).square().mean(-1).cpu().numpy())
            persistence.append(np.square(wz[:,0]-wz[:,h]).mean(-1))
    errors=np.stack(errors,1);persistence=np.stack(persistence,1)
    scale=float(np.var(z[train].astype(np.float64),axis=0,ddof=1).mean())
    if scale<=0: raise ValueError('collapsed target variance')
    result={'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'dataset':str(source),'sample_seed':20261106,'probe_split_sizes':[len(x) for x in (train,val,test)],
            'probe_scope':'episode-held-out probe from encoder training collection; not encoder-OOD',
            'probes':probes,'embedding_std_mean':float(np.std(z[test],axis=0,ddof=1).mean()),
            'participation_ratio':rank,'normalization_variance':scale,
            'rollout_initial_observations':1,'normalization':'mean per-dimension variance on probe-training frames',
            'rollouts':{str(h):{'simulator_steps':h*5,'normalized_mse':float(errors[:,h-1].mean()/scale),
                        'persistence_normalized_mse':float(persistence[:,h-1].mean()/scale),
                        'error_over_persistence':float(errors[:,h-1].mean()/persistence[:,h-1].mean())}
                        for h in [1,3,5,10]}}
    np.savez_compressed(out.with_suffix('.npz'),frame_ids=ids,split=splits,latents=z,window_starts=starts,
                        window_episodes=weps,rollout_mse=errors,persistence_mse=persistence,**factors)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
