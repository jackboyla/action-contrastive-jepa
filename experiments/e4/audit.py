"""E4 initialization, split, gradient and representation evidence."""
import hashlib
import json
import types

import torch


def tensor_hash(tensor):
    t = torch.as_tensor(tensor).detach().cpu().contiguous()
    h = hashlib.sha256(str((str(t.dtype), list(t.shape))).encode())
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def model_hash(model):
    return hashlib.sha256(json.dumps({k: tensor_hash(v) for k, v in
        sorted(model.state_dict().items())}, sort_keys=True).encode()).hexdigest()


def attach_audit(module, train_set, val_set, generator, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = dict(model_sha256=model_hash(module.model),
                    train_indices_sha256=tensor_hash(train_set.indices),
                    val_indices_sha256=tensor_hash(val_set.indices),
                    loader_rng_sha256=tensor_hash(generator.get_state()),
                    torch_rng_sha256=tensor_hash(torch.get_rng_state()),
                    train_clips=len(train_set), val_clips=len(val_set))
    path.with_name('initialization.json').write_text(json.dumps(identity, indent=2)+'\n')
    print('E4 INITIALIZATION', json.dumps(identity, sort_keys=True), flush=True)
    state = dict(backward_checks=0, forward_checks=0, nonfinite=False, diagnostics={})
    forward = module.forward

    def flush():
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
        temp.replace(path)

    def audited_forward(self, *args, **kwargs):
        out = forward(*args, **kwargs)
        scalars = {k: float(v.detach()) for k, v in out.items()
                   if torch.is_tensor(v) and v.numel() == 1}
        import math
        if not all(math.isfinite(v) for v in scalars.values()):
            state['nonfinite'] = True
            flush()
            raise FloatingPointError('Nonfinite training/validation scalar')
        state['forward_checks'] += 1
        for k in ('loss', 'forward_loss', 'inverse_loss', 'variance_reg_loss',
                  'emb_std', 'effective_rank', 'inverse_margin', 'inverse_baseline'):
            if k in scalars:
                series = state['diagnostics'].setdefault(k, dict(first=scalars[k],
                    min=scalars[k], max=scalars[k], last=scalars[k]))
                series.update(min=min(series['min'], scalars[k]),
                              max=max(series['max'], scalars[k]), last=scalars[k])
        if state['forward_checks'] % 100 == 0:
            with path.with_name('diagnostics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=int(self.global_step),
                    stage=kwargs.get('stage'), **scalars), allow_nan=False)+'\n')
            flush()
        return out

    def after_backward(self):
        norms = [p.grad.detach().float().norm() for p in self.model.parameters()
                 if p.grad is not None]
        total = torch.stack(norms).norm()
        if not bool(torch.isfinite(total)):
            state['nonfinite'] = True
            flush()
            raise FloatingPointError('Nonfinite gradient norm')
        self._last_grad_norm = float(total)
        state['backward_checks'] += 1

    module.forward = types.MethodType(audited_forward, module)
    module.after_manual_backward = types.MethodType(after_backward, module)
    module._e4_flush = flush
