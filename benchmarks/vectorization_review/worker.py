import argparse
import gc
import importlib
import json
from pathlib import Path
import sys
import time
import tracemalloc
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bench

parser=argparse.ArgumentParser()
parser.add_argument('root')
parser.add_argument('--verify')
parser.add_argument('--prefixes')
args=parser.parse_args()
sys.path.insert(0,args.root)
import skfuzzy
assert Path(skfuzzy.__file__).resolve().is_relative_to(Path(args.root).resolve())
if args.verify:
    rows=bench.cases()
    prefixes=json.loads(args.prefixes)
    selected=[dict(r,timing=False) for r in rows if any(r['id'].startswith(p) for p in prefixes)]
    bench.cases=lambda:selected
    bench.worker(args.root,args.verify,0)
    raise SystemExit()
warnings.simplefilter('ignore')
fixtures={r['id']:r for r in bench.cases()}
functions={}
counts={}
for line in sys.stdin:
    identifier=json.loads(line)
    case=fixtures[identifier]
    if identifier not in functions:
        if case['module']=='_control':fn=bench.controller(case['name'])
        elif case['module']=='_automf':fn=bench.automatic_memberships
        elif case['module']=='_scheduler':fn=bench.scheduler(case['kwargs']['size'])
        elif case['module']=='_fuzz':fn=bench.fuzz_many()
        else:fn=getattr(importlib.import_module(case['module']),case['name'])
        functions[identifier]=fn
    fn=functions[identifier]
    def call():return fn(*case['args'],**case['kwargs'])
    first=identifier not in counts
    if first:
        call()
        count=1
        while True:
            start=time.perf_counter_ns()
            for _ in range(count):call()
            if time.perf_counter_ns()-start>=20_000_000 or count>=16384:break
            count*=2
        counts[identifier]=count
    count=counts[identifier]
    start=time.perf_counter_ns()
    for _ in range(count):call()
    payload=dict(milliseconds=(time.perf_counter_ns()-start)/1e6/count,calls=count)
    if first:
        gc.collect()
        tracemalloc.start()
        call()
        _,payload['peak_bytes']=tracemalloc.get_traced_memory()
        tracemalloc.stop()
    print(json.dumps(payload),flush=True)
