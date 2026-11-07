import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from experiments.e4 import runner
from experiments.e4.audit import attach_audit


def test_frozen_pairs_differ_only_by_variance_and_paths(tmp_path):
    OmegaConf.register_new_resolver('eval',eval,replace=True)
    configs=[]
    for job in runner.jobs():
        cmd=runner.commands(job,tmp_path/job['name'],tmp_path)['train']
        with initialize_config_dir(config_dir=str(Path('config/train').resolve()),version_base=None):
            c=OmegaConf.to_container(compose(config_name=cmd[2],overrides=cmd[3:]),resolve=True)
        assert c['trainer']['max_epochs']==10 and c['runtime']['stop_after_epoch'] is None
        assert c['resume']['mode']=='never' and c['loss']['sigreg']['weight']==0
        assert c['loss']['masked']['inverse_weight']==1
        c.pop('subdir'); c['runtime'].pop('audit_path'); c['wandb']['config'].pop('id')
        c['loss']['variance_reg'].pop('weight'); configs.append(c)
    for i in range(0,6,2):
        assert configs[i]==configs[i+1]
    assert runner.project([dict(elapsed_seconds=36000)]*2)>runner.P['gpu_seconds_cap']


def test_audit_runs_in_real_manual_optimization_and_catches_bad_gradient(tmp_path):
    import lightning as pl
    import stable_pretraining as spt
    x=torch.arange(8,dtype=torch.float32).reshape(-1,1)
    dataset=[{'x':row} for row in x]
    subset=torch.utils.data.Subset(dataset,list(range(8)))
    def forward(self,batch,stage):
        loss=self.model(batch['x']).square().mean()
        return {'loss':loss,'emb_std':loss.detach()}
    model=spt.Module(model=torch.nn.Linear(1,1), forward=forward,
                     optim={'model_opt':{'modules':'model','optimizer':{'type':'SGD','lr':.01}}})
    attach_audit(model,subset,subset,torch.Generator().manual_seed(3),tmp_path/'finite.json')
    trainer=pl.Trainer(accelerator='cpu',max_epochs=1,logger=False,enable_checkpointing=False,
                       enable_progress_bar=False,enable_model_summary=False)
    trainer.fit(model,train_dataloaders=torch.utils.data.DataLoader(dataset,batch_size=2))
    model._e4_flush()
    state=json.loads((tmp_path/'finite.json').read_text())
    assert state['backward_checks']==4 and not state['nonfinite']
    parameter = next(model.model.parameters())
    parameter.grad = torch.full_like(parameter,float('nan'))
    with pytest.raises(FloatingPointError):
        model.after_manual_backward()


@pytest.mark.parametrize('interruption',[None,'completion','dispatch','budget'])
def test_supervisor_retains_receipts_and_gate_after_preemption(tmp_path,monkeypatch,interruption):
    import ast
    import time
    module=ast.parse(Path('modal_app.py').read_text())
    fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='run_e4_supervisor')
    fn.decorator_list=[]
    calls=[]
    interrupted=[False]
    class Preempted(BaseException):
        pass
    class Call:
        def __init__(self,job):
            self.job=job; self.hydrated=True
        @property
        def object_id(self):
            if not self.hydrated:
                raise AttributeError('unhydrated FunctionCall.from_id')
            return self.job['name']
        def get(self,timeout):
            if interruption=='completion' and self is calls[1] and not interrupted[0]:
                interrupted[0]=True; raise Preempted()
            return dict(job=self.job,elapsed_seconds=36000 if interruption=='budget' else 100,
                        source_sha256={},runtime_source_sha256={},packages_sha256='same',gpu_name='same',
                        training={'initialization':{}},evaluation={'tasks':{}})
        def cancel(self,**kwargs):
            pass
    def spawn(job,**kwargs):
        call=Call(job);calls.append(call)
        if interruption=='dispatch' and not interrupted[0]:
            interrupted[0]=True; raise Preempted()
        return call
    def from_id(ident):
        call=next(c for c in calls if c.job['name']==ident)
        call.hydrated=False
        return call
    def commit():
        path=tmp_path/'e4/test/status.json'
        if interruption=='budget' and path.exists() and not interrupted[0]:
            if len(json.loads(path.read_text())['completed'])==2:
                interrupted[0]=True; raise Preempted()
    monkeypatch.setattr(runner,'preflight',lambda *args:{})
    namespace=dict(STABLEWM_HOME=tmp_path,REPO_DIR=Path.cwd(),time=time,
                   subprocess=SimpleNamespace(run=lambda *a,**k:None),
                   volume=SimpleNamespace(reload=lambda:None,commit=commit),
                   modal=SimpleNamespace(current_function_call_id=lambda:'owner',
                       FunctionCall=SimpleNamespace(from_id=from_id)),
                   run_e4_job=SimpleNamespace(spawn=spawn))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'modal_app.py','exec'),namespace)
    run=namespace['run_e4_supervisor']
    if interruption:
        with pytest.raises(Preempted):run('test')
    status=run('test')
    if interruption=='dispatch':
        assert status['state']=='failed' and len(calls)==1
    elif interruption=='budget':
        assert status['state']=='budget_blocked' and len(calls)==2
    else:
        assert status['state']=='complete' and len(calls)==6 and len(status['completed'])==6
        assert run('test')==status and len(calls)==6


def test_worker_deadline_and_existing_output_fail_closed(tmp_path,monkeypatch):
    import sys,time,os
    monkeypatch.setenv('PATH',str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'])
    script=tmp_path/'wait.py'
    script.write_text('import time\ntime.sleep(60)\n')
    started=time.monotonic()
    with pytest.raises(RuntimeError,match='deadline'):
        runner.run_command([str(script)],tmp_path,'wait',time.time()+.1)
    assert time.monotonic()-started<10
    with pytest.raises(FileExistsError):
        runner.run_job(runner.jobs()[0],tmp_path,tmp_path,time.time()+60,lambda:None)


def test_installed_modal_from_id_is_lazy():
    import modal
    call=modal.FunctionCall.from_id('fc-1234567890123456789012')
    with pytest.raises(AttributeError,match='unhydrated'):
        _ = call.object_id
    assert callable(call.get) and callable(call.cancel)
