"""Execute exactly one campaign job from a job matrix, with fresh retrieval and provenance.

The adapter imports the runner's Click command only after replacing the shared
instance loader with one that bypasses the on-disk slot cache, so every job
retrieves its candidates afresh on the GPU and records what it loaded. Output
paths in the matrix are written with ${CAMPAIGN_ROOT} / ${MEMORY_GRIDS_ROOT}
placeholders, resolved from the environment (defaults: results/campaign and
results/memory_grids under the repository root).

  python scripts/exp4/run_job.py --validate                 # parse every job's arguments
  CUDA_VISIBLE_DEVICES=<uuid> python scripts/exp4/run_job.py --job default.amazon
"""
import argparse
import functools
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
PYTHON = sys.executable
GPU = os.environ.get('CUDA_VISIBLE_DEVICES', '')
ROOTS = {'CAMPAIGN_ROOT': os.environ.get('CAMPAIGN_ROOT', str(CODE / 'results' / 'campaign')),
         'MEMORY_GRIDS_ROOT': os.environ.get('MEMORY_GRIDS_ROOT', str(CODE / 'results' / 'memory_grids'))}


def resolve(job):
    """Expand the ${...} root placeholders of a matrix entry."""
    def expand(value):
        for key, root in ROOTS.items():
            value = value.replace('${' + key + '}', root)
        return value
    return {**job, 'output': expand(job['output']), 'args': [expand(a) for a in job['args']]}
THREADS = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS']


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def command(job):
    return importlib.import_module(job['module']).main


def validate(job):
    cli = command(job)
    # Click consumes the mutable argument list during parsing.
    with cli.make_context(job['module'], list(job['args'])) as context:
        return dict(context.params)


def dispatch(job, expected, metadata=None):
    cli = command(job)
    with cli.make_context(job['module'], list(job['args'])) as context:
        if dict(context.params) != expected:
            raise RuntimeError('Execution parameters differ from validated parameters')
        if metadata is not None:
            dump(metadata / 'execution_params.json', context.params)
        return cli.invoke(context)


def install_fresh_loader(directory, keep_instances=False):
    from gquery.alg.exp4 import core
    original = core.load_instances
    signature = inspect.signature(original)
    counter = 0

    @functools.wraps(original)
    def fresh(*args, **kwargs):
        nonlocal counter
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        bound.arguments['use_disk_cache'] = False
        batch, catalog = original(*bound.args, **bound.kwargs)
        from gquery.alg.exp4.protocol import write_manifest
        records = getattr(batch, 'requested_records', batch)
        counter += 1
        config = {k: v for k, v in bound.arguments.items() if k != 'items_cache'}
        config['supplied_items_cache'] = bound.arguments['items_cache'] is not None
        write_manifest(directory / f'load-{counter:03d}', config, batch)
        dump(directory / f'load-{counter:03d}' / 'inputs.json', [
            {'query_id': i.query.id, 'budget': i.query.budget,
             'n_slots': i.n_slots, 'candidate_counts': [len(s) for s in i.slots],
             'min_feasible_cost': i.min_feasible_cost,
             'status': core.instance_status(i, bound.arguments['alpha'])} for i in records])
        if keep_instances:
            import pickle
            (directory / f'load-{counter:03d}' / 'instances.pkl').write_bytes(
                pickle.dumps(list(records), protocol=pickle.HIGHEST_PROTOCOL))
        assert all(getattr(i, 'retrieval_source', None) == 'fresh_retrieval' for i in records)
        return batch, catalog

    core.load_instances = fresh
    return fresh


def worker(job, metadata):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one GPU before launching a job')
    from gquery.alg.exp4.protocol import write_manifest, sha256
    # Must precede imports of every runner and sensitivity dispatch helper.
    install_fresh_loader(metadata, job.get('pilot', False))
    params = validate(job)
    write_manifest(metadata / 'source', {'job': job, 'actual_click_params': params})
    dump(metadata / 'environment.json', {
        'pid': os.getpid(), 'parent_pid': os.getppid(), 'cwd': os.getcwd(),
        'started_ns': time.time_ns(), 'adapter_sha256': sha256(__file__),
        'matrix_sha256': sha256(MATRIX),
        'repository_commit': subprocess.check_output(
            ['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip(),
        'environment': {k: os.environ.get(k) for k in THREADS + ['CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PYTHONHASHSEED']},
        'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
        'logical_device': 'cuda:0', 'physical_gpu_uuid': GPU,
        'visible_device_name': torch.cuda.get_device_name(0),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'cpu_count': os.cpu_count(),
        'warmup': 'No implicit warmup; include first query startup in retrieval',
        'thread_pools': __import__('threadpoolctl').threadpool_info(),
        'thread_limit_scope': 'Configured environment and Torch; bundled HiGHS thread count is not verified by threadpoolctl',
        'method_order': 'Unmodified order of the invoked runner; one job at a time',
        'host_gpu_before': subprocess.run(['nvidia-smi'], text=True, capture_output=True).stdout,
    })
    dispatch(job, params, metadata)


def launch(job, matrix_path):
    import fcntl
    # CAMPAIGN_LOCK names a second lock only for an explicitly documented concurrent queue on another GPU
    lock = Path(os.environ.get('CAMPAIGN_LOCK', HERE / 'campaign.lock')).open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = Path(job['output'])
    metadata = output.parent / (output.name + '.job')
    if output.exists() or metadata.exists():
        raise FileExistsError(f'Refusing reuse of {output} or {metadata}')
    metadata.mkdir(parents=True)
    work = metadata / 'work'
    work.mkdir()
    env = os.environ.copy()
    env.update({k: '1' for k in THREADS})
    env.update(PYTHONPATH=str(CODE), PYTHONHASHSEED='0')
    if GPU:
        env['CUDA_VISIBLE_DEVICES'] = GPU
    argv = [PYTHON, str(Path(__file__).resolve()), '--matrix', str(matrix_path),
            '--job', job['id'], '--worker']
    record = {'job': job, 'command': argv, 'start_ns': time.time_ns(),
              'log': str(metadata / 'stdout.log'), 'status': 'starting'}
    dump(metadata / 'process.json', record)
    with (metadata / 'stdout.log').open('x') as log:
        # The worker retains the lock if this controller exits unexpectedly.
        proc = subprocess.Popen(argv, cwd=work, env=env, stdout=log,
                                stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
        record.update(pid=proc.pid, controller_pid=os.getpid(), status='running')
        dump(metadata / 'process.json', record)
        print(json.dumps(record), flush=True)
        code = proc.wait()
    record.update(exit_code=code, end_ns=time.time_ns(), status='completed' if code == 0 else 'failed')
    dump(metadata / 'process.json', record)
    print(json.dumps(record), flush=True)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', type=Path, default=HERE / 'jobs.json')
    parser.add_argument('--job')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--print-output', action='store_true', help='print the resolved output directory of --job and exit')
    args = parser.parse_args()
    MATRIX = args.matrix.resolve()
    jobs = json.loads(MATRIX.read_text())['jobs']
    if args.validate:
        results = {j['id']: validate(resolve(j)) for j in jobs}
        dump(HERE / ('pilot_cli_validation.json' if 'pilot' in MATRIX.name else 'cli_validation.json'), results)
        print(f'Validated {len(results)} real Click command configurations')
    else:
        job = resolve(next(j for j in jobs if j['id'] == args.job))
        if args.print_output:
            print(job['output'])
        elif args.worker:
            worker(job, Path(job['output']).parent / (Path(job['output']).name + '.job'))
        else:
            sys.exit(launch(job, MATRIX))
