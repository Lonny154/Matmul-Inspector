#!/usr/bin/env python3
"""Check controlled CUDA kernels in cuobjdump PTX; no GPU is required.

This checks instruction classes, not exact counts or compiler-specific registers.
An uninspectable image is unverified, never a successful no-FMA claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def inspect_ptx(text):
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    images = []
    for entry in re.finditer(r'\.entry\s+(\w+)\s*\(', text):
        name = entry.group(1)
        if name not in ('mi_naive_fma', 'mi_naive_no_fma', 'mi_naive_reordered'):
            continue
        start = text.find('{', entry.end())
        if start < 0:
            raise ValueError('PTX entry has no body')
        depth, end = 1, start + 1
        while depth and end < len(text):
            depth += (text[end] == '{') - (text[end] == '}')
            end += 1
        if depth:
            raise ValueError("Unterminated PTX entry")
        body = text[start:end]
        counts = {op: len(re.findall(r'\b' + op + r'(?:\.[\w]+)*\.f32\b', body))
                  for op in ('fma', 'mul', 'add', 'mad')}
        if counts['mad']:
            raise ValueError('Ambiguous FP32 mad instruction requires manual inspection')
        if re.search(r'\bcall(?:\.|\s)', body):
            raise ValueError('Out-of-line PTX calls require manual inspection')
        if name == 'mi_naive_no_fma':
            valid = counts['fma'] == 0 and counts['mul'] > 0 and counts['add'] > 0
        else:
            valid = counts['fma'] > 0 and counts['mul'] == 0
        if not valid:
            raise ValueError(f'Unexpected FP32 arithmetic in {name}: {counts}')
        images.append(dict(kernel=name, **counts))
    if {image['kernel'] for image in images} != {'mi_naive_fma', 'mi_naive_no_fma', 'mi_naive_reordered'}:
        raise ValueError('Missing controlled kernel PTX; compile with a virtual/PTX architecture target')
    return images


def write_changed(path, contents):
    if path is not None and (not path.exists() or path.read_text() != contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--cuobjdump', default='cuobjdump')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--header', type=Path)
    parser.add_argument('--allow-unverified', action='store_true', help='Leave ordinary kernels usable when inspection is unavailable')
    args = parser.parse_args()
    report = {'verified': False, 'method': 'cuobjdump --dump-ptx; FP32 fma vs separate mul/add; all embedded PTX images',
              'binary_sha256': 'unknown'}
    try:
        report['binary_sha256'] = hashlib.sha256(args.binary.read_bytes()).hexdigest()
        output = subprocess.run([args.cuobjdump, '--dump-ptx', str(args.binary)],
                                capture_output=True, text=True, check=True, timeout=60)
        report['images'] = inspect_ptx(output.stdout)
        report['verified'] = True
        report['details'] = json.dumps(report['images'], sort_keys=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report['details'] = str(error)
    write_changed(args.report, json.dumps(report, indent=2) + '\n')
    header = '#pragma once\nnamespace fp_verification {\n'
    header += 'inline constexpr bool verified = ' + str(report['verified']).lower() + ';\n'
    for key in ('method', 'binary_sha256', 'details'):
        header += f'inline constexpr const char* {key} = {json.dumps(report[key], ensure_ascii=True)};\n'
    write_changed(args.header, header + '}\n')
    print('Controlled CUDA FP verification: ' + ('PASS' if report['verified'] else 'UNVERIFIED: ' + report['details']))
    return 0 if report['verified'] or args.allow_unverified else 1


if __name__ == '__main__':
    raise SystemExit(main())
