#!/usr/bin/env python3
"""Replay the input/configuration in metadata.json using an already built binary."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from operations import identity


def command(metadata, executable, output):
    if metadata.get('schema_version') not in (1, 2, 3):
        raise ValueError('Unsupported metadata schema')
    config = metadata['config']
    generators = {'random': 'lcg32-v1', 'cancellation': 'cancellation-v1', 'fma-sensitive': 'fma-sensitive-v1'}
    input_mode = config.get('input', 'random')
    if config['generator'] != generators.get(input_mode) or config['dtype'] != 'float32':
        raise ValueError('Unsupported generator or dtype')
    mode = config['mode']
    if mode not in ('compare', 'benchmark', 'crossover'):
        raise ValueError('Unsupported mode')
    args = [str(executable), mode]
    shapes = config['shapes']
    operation = identity(config)
    if operation != 'matmul':
        if config.get('reduction_block_size') != 256:
            raise ValueError('Unsupported reduction block size')
        if not shapes or any(s['M'] != 1 or s['N'] != 1 or s['K'] <= 0 for s in shapes):
            raise ValueError('Invalid vector shapes')
        args += ['--operation', operation, '--sizes', ','.join(str(s['K']) for s in shapes)]
    elif shapes and all(s['M'] == s['N'] == s['K'] for s in shapes):
        args += ['--sizes', ','.join(str(s['M']) for s in shapes)]
    elif len(shapes) == 1:
        for key in ('M', 'N', 'K'):
            args += ['--' + key.lower(), str(shapes[0][key])]
    else:
        raise ValueError('Cannot represent these shapes in one CLI invocation')
    for flag, key in (('--seed', 'seed'), ('--seed-b', 'seed_b'), ('--atol', 'atol'), ('--rtol', 'rtol')):
        args += [flag, str(config[key])]
    args += ['--reference', config['reference'], '--candidate', config['candidate']]
    if metadata['schema_version'] >= 2:
        args += ['--input', input_mode, '--max-mismatches', str(config['max_mismatches'])]
    if mode in ('benchmark', 'crossover'):
        args += ['--warmups', str(config['warmups']), '--iterations', str(config['iterations'])]
    if mode in ('benchmark', 'crossover'):
        for field in ('trials', 'bootstrap_samples', 'confidence_level', 'bootstrap_seed'):
            if field in config:
                args += ['--' + field.replace('_', '-'), str(config[field])]
        if 'percentiles' in config:
            args += ['--percentiles', ','.join(map(str, config['percentiles']))]
    if config.get('save_output', False):
        args += ['--save-output']
    return args + ['--output', str(output)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata', type=Path)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        metadata = json.loads(args.metadata.read_text())
        invocation = command(metadata, args.executable.resolve(), args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(shlex.join(invocation), flush=True)
    if metadata.get('git_dirty') is not False:
        print('WARNING: Original source was dirty or unknown; configuration replay cannot reconstruct it.', flush=True)
    return 0 if args.dry_run else subprocess.run(invocation).returncode


if __name__ == '__main__':
    raise SystemExit(main())
