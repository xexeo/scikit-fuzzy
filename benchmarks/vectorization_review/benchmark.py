"""Reproducible isolated-process differential checks and speed measurements.

Run with Python from a scientific environment: python optimization/benchmark.py
Both reference commits are ancestors of main; no network access is required.
"""
import argparse
import copy
import gc
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
import warnings
import zipfile

BASELINE = '7777a10abe4dbd7e508ae1c810f05017cae14c54'
UPSTREAM = 'be7e344f3ce65845e39922aa5fc1af7412d01944'
ROOT = Path(__file__).resolve().parents[2]


def cases():  # noqa: C901  Fixture enumeration has no production control flow.
    import numpy as np
    rng = np.random.default_rng(20261119)
    rows = []

    def add(identifier, module, name, args, kwargs=None, timing=False):
        rows.append(dict(id=identifier, module='skfuzzy.' + module, name=name,
                         args=args, kwargs=kwargs or {}, timing=timing))

    for dtype in [np.float32, np.float64]:
        for n in [1, 4, 101, 10001]:
            x = np.linspace(-5, 10, n, dtype=dtype)
            mf = rng.random(n).astype(dtype)
            for name in ['centroid', 'bisector']:
                add(f'{name}/{dtype.__name__}/{n}', 'defuzzify.defuzz', name,
                    [x, mf], timing=dtype == np.float64 and n in [101, 10001])
        for seed in range(100):
            x = np.cumsum(rng.random(25).astype(dtype))
            mf = rng.random(25).astype(dtype)
            mf[rng.random(25) < 0.4] = 0
            for name in ['centroid', 'bisector']:
                add(f'{name}/random/{dtype.__name__}/{seed}', 'defuzzify.defuzz', name, [x, mf])
    for name in ['centroid', 'bisector']:
        for label, x, mf in [
                ('empty', np.array([]), np.array([])),
                ('zero', np.arange(5.), np.zeros(5)),
                ('repeated', np.array([0., 0., 1., 2., 2.]), np.array([0., .3, 1., .2, 0.])),
                ('nan', np.arange(4.), np.array([0., 1., np.nan, 0.])),
                ('signed', np.arange(4.), np.array([0., -1., 1., 0.])),
                ('integers', np.arange(4), np.array([0, 1, 2, 0])),
                ('large_offset', 1e12 + np.arange(5.), np.array([0., .2, 1., .4, 0.]))]:
            add(f'{name}/edge/{label}', 'defuzzify.defuzz', name, [x, mf])
    for n in [17, 101, 10001]:
        x = np.linspace(0, 10, n)
        mf = np.maximum(0., 1. - np.abs(x - 5.) / 4.)
        for level in [0., .2, .5, 1.]:
            add(f'boundaries/{n}/{level}', 'defuzzify.defuzz', 'lambda_cut_boundaries',
                [x, mf, level], timing=level == .5 and n in [101, 10001])
        for levels in [11, 101]:
            add(f'cuts/{n}/{levels}', 'defuzzify.defuzz', 'lambda_cut_series',
                [x, mf, levels], timing=n == 10001 and levels == 101)
    for dtype in [np.float32, np.float64]:
        x = np.linspace(0, 10, 101, dtype=dtype)
        mf = np.maximum(0, 1 - np.abs(x - 5) / 4).astype(dtype)
        for level in [0., .25, .5, 1.]:
            add(f'boundaries/{dtype.__name__}/{level}', 'defuzzify.defuzz', 'lambda_cut_boundaries', [x, mf, level])
    add('bisector/analytic_gap', 'defuzzify.defuzz', 'bisector',
        [np.arange(7.), np.array([0., 1., 0., 0., 0., 1., 1.])])
    for n in [101, 1001]:
        x = np.linspace(1, 5, n)
        y = np.linspace(2, 8, n)
        a = np.maximum(0, 1 - np.abs(x - 3.) / 2.)
        b = np.maximum(0, 1 - np.abs(y - 5.) / 3.)
        for name in ['dsw_add', 'dsw_sub', 'dsw_mult', 'dsw_div']:
            add(f'{name}/{n}/101', 'intervals.intervalops', name, [x, a, y, b, 101], timing=n == 1001)
    for name in ['maxmin_composition', 'maxprod_composition']:
        for m, n, p in [(1, 4, 1), (8, 16, 8), (64, 32, 64), (128, 64, 128)]:
            a, b = rng.random((m, n)), rng.random((n, p))
            add(f'{name}/{m}x{n}x{p}', 'fuzzymath.fuzzy_ops', name, [a, b], timing=m >= 8)
        for dtype in [np.float32, np.int16]:
            a, b = (rng.random((3, 5)) * 5).astype(dtype), (rng.random((5, 4)) * 5).astype(dtype)
            add(f'{name}/{dtype.__name__}', 'fuzzymath.fuzzy_ops', name, [a, b])
        add(f'{name}/nan', 'fuzzymath.fuzzy_ops', name,
            [np.array([[0.5, np.nan], [1., .2]]), np.array([[.4, 1.], [.8, .7]])])
        add(f'{name}/vectors', 'fuzzymath.fuzzy_ops', name, [np.array([.1, .3]), np.array([.6, .9])])
        add(f'{name}/empty', 'fuzzymath.fuzzy_ops', name, [np.empty((2, 0)), np.empty((0, 3))])
        add(f'{name}/zero_outputs', 'fuzzymath.fuzzy_ops', name, [np.empty((2, 0)), np.empty((0, 0))])
        for seed in range(20):
            a, b = rng.random((3, 9)), rng.random((9, 5))
            add(f'{name}/random/{seed}', 'fuzzymath.fuzzy_ops', name, [a, b])
    for name in ['cartadd', 'cartprod', 'classic_relation', 'relation_min', 'relation_product']:
        for n in [17, 501]:
            a, b = rng.random(n), rng.random(n + 3)
            add(f'{name}/{n}', 'fuzzymath.fuzzy_ops', name, [a, b], timing=True)
        add(f'{name}/nonfinite', 'fuzzymath.fuzzy_ops', name,
            [np.array([-.2, 1.2, np.nan]), np.array([0., 1., np.nan])])
    for name in ['inner_product', 'outer_product']:
        for n in [101, 100001]:
            add(f'{name}/{n}', 'fuzzymath.fuzzy_ops', name, [rng.random(n), rng.random(n)], timing=True)
        add(f'{name}/lists', 'fuzzymath.fuzzy_ops', name, [[.1, .3], [.5, .2]])
    for n in [100, 10000]:
        add(f'fire1d/{n}', 'filters.fire', 'fire1d', [rng.uniform(-1, 1, n)], timing=True)
    for n in [32, 256]:
        add(f'fire2d/{n}', 'filters.fire', 'fire2d', [rng.integers(30, 220, (n, n))], timing=True)
        add(f'defocus/{n}', 'image.imops', 'defocus_local_means', [rng.normal(size=(n, n))], timing=True)
    for seed in range(25):
        add(f'fire1d/random/{seed}', 'filters.fire', 'fire1d', [rng.uniform(-2, 2, 30)])
    for dtype in [np.float32, np.float64]:
        for c, n in [(3, 100), (32, 10000), (128, 2000)]:
            u = rng.random((c, n)).astype(dtype)
            u /= u.sum(axis=0)
            add(f'fpc/{dtype.__name__}/{c}x{n}', 'cluster._cmeans', '_fp_coeff', [u], timing=dtype == np.float64)
    for n in [100, 1000]:
        data = rng.normal(size=(3, n))
        init = rng.random((4, n))
        init /= init.sum(axis=0)
        add(f'cmeans/{n}', 'cluster._cmeans', 'cmeans', [data, 4, 2., 1e-6, 20], {'init': init}, timing=True)
        centers = rng.normal(size=(4, 3))
        add(f'cmeans_predict/{n}', 'cluster._cmeans', 'cmeans_predict',
            [data, centers, 2., 1e-6, 20], {'init': init}, timing=True)
    for dtype in [np.float32, np.float64]:
        for n in [101, 100001]:
            x = np.linspace(-3, 8, n, dtype=dtype)
            for name, parameters in [('trimf', [0., 2., 6.]), ('trapmf', [0., 2., 4., 6.])]:
                add(f'{name}/{dtype.__name__}/{n}', 'membership.generatemf', name,
                    [x, parameters], timing=dtype == np.float64)
        for parameters in [[1, 1, 1, 1], [0, 0, 2, 2], [0, 1, 1, 2]]:
            add('trapmf/edge/' + str(parameters), 'membership.generatemf', 'trapmf',
                [np.array([-1., 0., 1., 2., 3., np.nan]), parameters])
    for dtype in [np.float32, np.float64, np.int64]:
        wx = (rng.random((64, 2000)) * 3).astype(dtype)
        b = (rng.random(64) * 3).astype(dtype)
        add(f'sigmoid/{dtype.__name__}/64x2000', 'membership.generatemf', 'sigmoid', [wx, b], timing=dtype == np.float64)
    for name, x, y in [('nearest', np.arange(10001.), 4444.5), ('nearest_nan', np.array([1., np.nan]), 1.)]:
        add(name, 'membership.generatemf', '_nearest', [x, y], timing=name == 'nearest')
    for exponent in [-20., -2., .5, 2.]:
        add(f'normalize_power/{exponent}', 'cluster.normalize_columns', 'normalize_power_columns',
            [rng.random((32, 10000)), exponent], timing=True)
    for mode in ['centroid', 'bisector', 'mom', 'som', 'lom']:
        for size in [15, 501]:
            rows.append(dict(id=f'controller/{mode}/{size}', module='_control', name=mode,
                             args=[size], kwargs={}, timing=size == 501))
    for size in [9, 101]:
        rows.append(dict(id=f'automf/{size}/1001', module='_automf', name='automf',
                         args=[size], kwargs={}, timing=True))
    for size in [10, 100]:
        rows.append(dict(id=f'rule_scheduler/{size}', module='_scheduler', name='rule_scheduler',
                         args=[], kwargs={'size': size}, timing=True))
    for size in [32, 1001]:
        rows.append(dict(id=f'fuzz_many/101terms/{size}', module='_fuzz', name='fuzz',
                         args=[size], kwargs={}, timing=True))
    add('legacy_arange/20x30x40', 'image.arraypad', '_arange_ndarray',
        [np.zeros((20, 30, 40)), (20, 30, 40), 1], timing=True)
    for n in [31, 101]:
        x = np.arange(n, dtype=float)
        a = np.maximum(0, 1 - abs(x - n / 3.) / (n / 3.))
        b = np.maximum(0, 1 - abs(x - 2 * n / 3.) / (n / 3.))
        positive = x > 0
        for name in ['fuzzy_add', 'fuzzy_sub', 'fuzzy_mult', 'fuzzy_div', 'fuzzy_min']:
            args = [x, a, x, b] if name != 'fuzzy_div' else [x, a, x[positive], b[positive]]
            add(f'{name}/discrete/{n}', 'fuzzymath.fuzzy_ops', name, args, timing=True)
            if n == 31:
                add(f'{name}/alpha/{n}', 'fuzzymath.fuzzy_ops', name, [x, a, x, b],
                    {'method': 'alpha', 'n_levels': 30}, timing=True)
    return rows


def worker(root, output, repeat):
    sys.path.insert(0, str(root))
    import numpy as np
    import scipy
    import skfuzzy
    assert Path(skfuzzy.__file__).resolve().is_relative_to(Path(root).resolve())
    results = []
    for case in cases():
        if case['module'] == '_control':
            fn = controller(case['name'])
        elif case['module'] == '_automf':
            fn = automatic_memberships
        elif case['module'] == '_scheduler':
            fn = scheduler(case['kwargs']['size'])
        elif case['module'] == '_fuzz':
            fn = fuzz_many()
        else:
            fn = getattr(importlib.import_module(case['module']), case['name'])
        args, kwargs = copy.deepcopy(case['args']), copy.deepcopy(case['kwargs'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                value = fn(*args, **kwargs)
                error = None
            except Exception as exc:
                value = None
                error = {'type': type(exc).__name__, 'message': str(exc)}
        row = {'id': case['id'], 'value': value, 'error': error,
               'warnings': [w.category.__name__ for w in caught],
               'mutated_args': args if case['name'] == 'fire1d' else None}
        if error is None and case['name'] == 'bisector' and case['module'] != '_control':
            row['oracle'] = bisector_residual(case['args'][0], case['args'][1], value)
        if error is None and case['name'] == '_fp_coeff':
            u = case['args'][0].astype(float)
            expected = math.fsum((u * u).ravel()) / u.shape[1]
            row['oracle'] = dict(compensated_float64=expected, absolute_error=abs(float(value) - expected))
        if case['timing'] and error is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                def call():
                    return fn(*case['args'], **case['kwargs'])
                call()
                call()
                # Calibrate a group of calls so scheduler jitter does not
                # dominate microsecond kernels. Keep raw per-call samples.
                count = 1
                while True:
                    start = time.perf_counter_ns()
                    for _ in range(count):
                        call()
                    elapsed = (time.perf_counter_ns() - start) / 1e9
                    if elapsed >= .04 or count >= 65536:
                        break
                    count *= 2
                samples = []
                for _ in range(repeat):
                    start = time.perf_counter_ns()
                    for _ in range(count):
                        call()
                    samples.append((time.perf_counter_ns() - start) / 1e6 / count)
                gc.collect()
                tracemalloc.start()
                call()
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            row.update(milliseconds=samples, median_ms=statistics.median(samples),
                       calls_per_sample=count, tracked_peak_bytes=peak)
        results.append(row)
    payload = {'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'scipy': scipy.__version__, 'package': str(skfuzzy.__file__),
                              'platform': platform.platform(), 'processor': platform.processor()},
               'source_sha256': {str(p.relative_to(root)).replace('\\', '/'):
                                 hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(root) / 'skfuzzy').rglob('*.py')
                                 if 'tests' not in p.parts},
               'rows': results}
    Path(output).write_bytes(pickle.dumps(payload, protocol=5))
    print(str(root), len(results), 'cases', flush=True)


def controller(mode):
    import numpy as np
    from skfuzzy import control as ctrl
    a = ctrl.Antecedent(np.linspace(0, 10, 21), 'a')
    b = ctrl.Antecedent(np.linspace(0, 10, 21), 'b')
    out = ctrl.Consequent(np.linspace(0, 25, 51), 'out')
    for variable in [a, b, out]:
        variable.automf(3)
    out.defuzzify_method = mode
    rules = [ctrl.Rule(a['poor'] | b['poor'], out['poor']),
             ctrl.Rule(a['average'], out['average']),
             ctrl.Rule(a['good'] | b['good'], out['good'])]
    sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules), cache=False)

    def run(size):
        x = np.linspace(0, 10, size)
        sim.input['a'] = x
        sim.input['b'] = 5 + 4 * np.cos(x)
        sim.compute()
        return sim.output['out'].copy()
    return run


def automatic_memberships(number):
    import numpy as np
    from skfuzzy.control import Antecedent
    variable = Antecedent(np.linspace(0, 1, 1001), 'terms')
    names = ['term' + str(i) for i in range(number)]
    variable.automf(names=names)
    assert list(variable.terms) == names
    return np.stack([term.mf for term in variable.terms.values()])


def scheduler(size):
    import numpy as np
    from skfuzzy import control as ctrl
    a = ctrl.Antecedent(np.arange(11.), 'a')
    out = ctrl.Consequent(np.arange(11.), 'out')
    a.automf(3)
    out.automf(3)
    rules = [ctrl.Rule(a['average'], out['good'], label='rule' + str(i)) for i in range(size)]
    system = ctrl.ControlSystem(rules)

    def run(size):
        return tuple(rule.label for rule in system.rules)
    return run


def fuzz_many():
    import numpy as np
    from skfuzzy import control as ctrl
    from skfuzzy.control.controlsystem import CrispValueCalculator
    variable = ctrl.Antecedent(np.linspace(0, 1, 1001), 'many')
    variable.automf(names=['term' + str(i) for i in range(101)])
    out = ctrl.Consequent(np.arange(3.), 'out')
    out.automf(3)
    rule = ctrl.Rule(variable['term0'], out['poor'])
    sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem([rule]))
    calculator = CrispValueCalculator(variable, sim)

    def run(size):
        values = np.linspace(-.1, 1.1, size)
        calculator.fuzz(values)
        return np.stack([term.membership_value[sim] for term in variable.terms.values()])
    return run


def bisector_residual(x, mf, result):
    """Integrate the independent piecewise linear definition in float64."""
    import numpy as np
    x, mf = np.asarray(x, dtype=float), np.asarray(mf, dtype=float)
    if (x.size < 2 or not np.isfinite(x).all() or not np.isfinite(mf).all()
            or (mf < 0).any() or (np.diff(x) < 0).any() or not np.isfinite(result)):
        return None
    widths = np.diff(x)
    total = math.fsum(widths * (mf[:-1] + mf[1:]) / 2.)
    if total <= 0:
        return None
    pieces = []
    for i, width in enumerate(widths):
        if width == 0:
            continue
        partial = float(np.clip(result - x[i], 0, width))
        pieces.append(mf[i] * partial + (mf[i + 1] - mf[i]) * partial ** 2 / (2 * width))
    left = math.fsum(pieces)
    return dict(total_area=total, left_area=left, target=total / 2.,
                relative_residual=abs(left - total / 2.) / total)


def compare(a, b):
    import numpy as np
    if isinstance(a, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            return dict(exact=False, compatible=False, max_abs=None)
        children = [compare(x, y) for x, y in zip(a, b)]
        return dict(exact=all(x['exact'] for x in children), compatible=all(x['compatible'] for x in children),
                    max_abs=max((x['max_abs'] or 0 for x in children), default=0),
                    differing_components=[dict(index=i, **child) for i, child in enumerate(children) if not child['exact']])
    if a is None or b is None:
        return dict(exact=a is b, compatible=a is b, max_abs=0)
    aa, bb = np.asarray(a), np.asarray(b)
    if aa.shape != bb.shape or aa.dtype != bb.dtype:
        return dict(exact=False, compatible=False, max_abs=None)
    exact = aa.tobytes() == bb.tobytes() and type(a) is type(b)
    if aa.dtype.kind not in 'biufc':
        return dict(exact=exact, compatible=bool(np.array_equal(aa, bb)), max_abs=0)
    if aa.size == 0:
        return dict(exact=exact, compatible=True, max_abs=0)
    compatible = bool(np.allclose(aa, bb, rtol=2e-6 if aa.dtype == np.float32 else 2e-12,
                                  atol=2e-7 if aa.dtype == np.float32 else 2e-13, equal_nan=True))
    with np.errstate(all='ignore'):
        delta = np.abs(aa.astype(float) - bb.astype(float))
    finite = delta[np.isfinite(delta)]
    return dict(exact=exact, compatible=compatible, max_abs=float(finite.max()) if finite.size else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', nargs=2, metavar=('ROOT', 'OUTPUT'))
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--skip-upstream', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker, args.repeat)
        return
    cache = ROOT / '.benchmark-cache'
    cache.mkdir(exist_ok=True)
    output = ROOT / 'optimization/results'
    output.mkdir(exist_ok=True)
    roots = {'baseline': cache / 'baseline', 'candidate': ROOT}
    if not args.skip_upstream:
        roots['upstream'] = cache / 'upstream'
    for key, sha in [('baseline', BASELINE), ('upstream', UPSTREAM)]:
        if key not in roots:
            continue
        target = roots[key]
        if not target.exists():
            raw = subprocess.check_output(['git', 'archive', '--format=zip', sha], cwd=ROOT)
            zipfile.ZipFile(io.BytesIO(raw)).extractall(target)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MPLBACKEND='Agg', PYTHONHASHSEED='0')
    data = {}
    for key, root in roots.items():
        target = cache / (key + '.pkl')
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(root),
                        str(target), '--repeat', str(args.repeat)], check=True, env=env, cwd=cache)
        data[key] = pickle.loads(target.read_bytes())
    ref = {x['id']: x for x in data['baseline']['rows']}
    original = {x['id']: x for x in data.get('upstream', {}).get('rows', [])}
    comparisons = []
    for row in data['candidate']['rows']:
        base = ref[row['id']]
        if base['error'] or row['error']:
            comparison = dict(exact=base['error'] == row['error'],
                              compatible=(base['error'] or {}).get('type') == (row['error'] or {}).get('type'), max_abs=None)
        else:
            comparison = compare(base['value'], row['value'])
        comparison.update(id=row['id'], baseline_error=base['error'], candidate_error=row['error'],
                          warning_categories_equal=base['warnings'] == row['warnings'])
        if 'oracle' in row:
            comparison.update(baseline_oracle=base.get('oracle'), candidate_oracle=row['oracle'])
        if row['mutated_args'] is not None:
            comparison['mutation_equal'] = compare(base['mutated_args'], row['mutated_args'])
        if 'median_ms' in base and 'median_ms' in row:
            comparison.update(baseline_ms=base['median_ms'], candidate_ms=row['median_ms'],
                              speedup=base['median_ms'] / row['median_ms'],
                              baseline_samples_ms=base['milliseconds'], candidate_samples_ms=row['milliseconds'],
                              baseline_calls_per_sample=base['calls_per_sample'],
                              candidate_calls_per_sample=row['calls_per_sample'],
                              baseline_peak_bytes=base['tracked_peak_bytes'], candidate_peak_bytes=row['tracked_peak_bytes'])
        old = original.get(row['id'], {})
        if 'median_ms' in old and 'median_ms' in row:
            comparison.update(upstream_ms=old['median_ms'], speedup_vs_upstream=old['median_ms'] / row['median_ms'])
        comparisons.append(comparison)
    report = {'baseline_commit': BASELINE, 'upstream_commit': UPSTREAM,
              'repeat': args.repeat, 'environment': data['candidate']['environment'],
              'source_sha256': data['candidate']['source_sha256'],
              'case_count': len(comparisons), 'exact': sum(x['exact'] for x in comparisons),
              'compatible': sum(x['compatible'] for x in comparisons), 'comparisons': comparisons}
    (output / 'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'comparisons'}, indent=2))
    print('Nonexact', [(x['id'], x['max_abs']) for x in comparisons if not x['exact']])
    print('Incompatible', [x for x in comparisons if not x['compatible']])


if __name__ == '__main__':
    main()
