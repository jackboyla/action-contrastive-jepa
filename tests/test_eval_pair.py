import fcntl
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('pair', Path(__file__).parents[1]/'scripts/run_eval_pair.py')
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)


class PairTest(unittest.TestCase):
    def test_failure_keeps_lock_until_sibling_exits_and_retains_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            jobs=[]
            for index, code in enumerate([0, 3]):
                command="import time; print('CEM solve time: 1', flush=True); time.sleep(%s); raise SystemExit(%s)" % (.25 if index==0 else .02, code)
                jobs.append(dict(argv=[sys.executable,'-c',command],log=str(root/f'{index}.log'),output=str(root/f'{index}.out')))
            config=dict(lock=str(root/'gpu.lock'),cwd=directory,env=dict(os.environ),jobs=jobs,report=str(root/'receipt.json'))
            checks=[]
            def sample():
                with open(config['lock'],'a') as lock:
                    try:
                        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        checks.append(True)
                    else:
                        self.fail('lock released during trial')
                return dict(gpu_used_mib=4000,gpu_free_mib=28000,ram_available_mib=32000)
            result=pair.execute(config,sample=sample,locate=lambda pid:{'gpu0_compute_and_graphics':True},interval=.01,stagger=.03)
            self.assertEqual(result['mode'],'parallel')
            self.assertEqual(result['status'],'failed')
            self.assertEqual([j['exit_code'] for j in result['jobs']],[0,3])
            self.assertGreater(result['ended'],result['jobs'][0]['ended']-.001)
            self.assertTrue(checks)
            with open(config['lock'],'a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(AssertionError):
                pair.execute(config)

    def test_headroom_requires_gpu_and_ram_reserves(self):
        self.assertTrue(pair.headroom(dict(gpu_free_mib=12288,ram_available_mib=16384)))
        self.assertFalse(pair.headroom(dict(gpu_free_mib=12287,ram_available_mib=16384)))
        self.assertFalse(pair.headroom(dict(gpu_free_mib=12288,ram_available_mib=16383)))

    def test_low_headroom_runs_serially(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            jobs=[dict(argv=[sys.executable,'-c',"import time; print('CEM solve time: 1',flush=True); time.sleep(.1)"],log=str(root/f'{i}.log'),output=str(root/f'{i}.out')) for i in range(2)]
            config=dict(lock=str(root/'lock'),cwd=directory,env=dict(os.environ),jobs=jobs,report=str(root/'receipt.json'))
            result=pair.execute(config,sample=lambda:dict(gpu_used_mib=24000,gpu_free_mib=8000,ram_available_mib=32000),locate=lambda pid:{'gpu0_compute_and_graphics':True},interval=.01,stagger=.02)
            self.assertEqual(result['mode'],'serial')
            self.assertEqual(result['status'],'complete')
            self.assertGreaterEqual(result['jobs'][1]['launched'],result['jobs'][0]['ended'])
