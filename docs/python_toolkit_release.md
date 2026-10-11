# Python toolkit v0.1.0 release guide

The `matmul-inspector` Python distribution packages the existing PTX, SASS, NKI,
and unified reporting tools. It is deliberately CPU-only and standard-library-only
at runtime. It does not package or build the native C++/CUDA benchmark executable.

## Package layout

```text
src/matmul_inspector/
  __init__.py
  cli.py
  ptx.py
  sass.py
  nki.py
  reporting.py
```

The files under `scripts/` remain compatibility entry points for source checkouts
and delegate to these package modules. There is one implementation of each
analysis tool; script and installed invocation share the same functions and JSON
schemas.

## Installation and commands

The public installation target is:

```sh
pip install matmul-inspector
```

The project is not yet published. For prerelease testing, build locally and install
one of the generated archives:

```sh
python3 -m pip install build
python3 -m build
python3 -m venv /tmp/matmul-inspector-test
/tmp/matmul-inspector-test/bin/python -m pip install --no-deps dist/*.whl
/tmp/matmul-inspector-test/bin/matmul-inspector --version
```

The installed dispatcher supports:

```text
matmul-inspector ptx inspect ...
matmul-inspector ptx compare ...
matmul-inspector sass inspect ...
matmul-inspector sass compare ...
matmul-inspector nki ...
matmul-inspector report ...
```

`matmul-inspector-python` is an alias for the same dispatcher.

## Native executable name collision

The CMake project continues to build its native executable as `matmul-inspector`.
The Python distribution also uses that conventional console-script name. No native
target is renamed or replaced; shell `PATH` ordering determines which executable
runs when both are installed.

Use `command -v matmul-inspector` to inspect resolution. Use
`matmul-inspector-python` when the active environment is expected to contain the
Python toolkit. That alias is not globally authoritative: another environment can
provide an executable with the same name, and an inactive virtual environment will
not normally place the intended alias on `PATH`. Activating the expected environment
or using its absolute `bin/matmul-inspector-python` path removes the ambiguity.

## Hardware boundary

Installed analysis commands consume saved artifacts and require no GPU, CUDA
toolkit, AWS Neuron SDK, Trainium device, native compiler, or CMake. Generating PTX,
SASS, NEFF, compiler statistics, or benchmark measurements remains a separate
hardware/toolchain workflow. The package does not add capture, profiling, or
benchmark orchestration.

## Distribution validation

Normal packaging CI builds wheel and sdist archives, audits archive contents, and
installs them in clean virtual environments. Smoke tests run from a temporary
directory outside the checkout with `PYTHONPATH` and `PYTHONHOME` removed. They
exercise version/help output plus minimal PTX, SASS, NKI, and unified-report
workflows. CI does not publish packages and needs no accelerator hardware.
The configured Linux matrix is Python 3.10–3.13; local clean-install validation
currently covers Python 3.13, so the complete support claim remains provisional
until the matrix succeeds.

The archives intentionally exclude native sources and build products, repository
scripts/tests, benchmark results, generated compiler artifacts, local context
files, and local environments. The sdist includes package sources, README, build
metadata, and setuptools-generated distribution metadata.

## v0.1.0 release checklist

- [x] Distribution name and version recorded as `matmul-inspector` 0.1.0.
- [x] Python 3.10–3.13 Linux build and clean-install CI defined.
- [x] Wheel and sdist content audit defined.
- [x] Runtime dependencies remain empty.
- [x] License the project under Apache License 2.0 and include `LICENSE` and
  `NOTICE` in both distribution formats.
- [ ] Confirm `matmul-inspector` availability through the PyPI publication workflow.
- [ ] Review distribution metadata and project URLs.
- [ ] Validate the candidate archives on TestPyPI.
- [ ] Publish the approved archives to PyPI.
- [ ] Create the approved `v0.1.0` Git tag and GitHub release.
- [ ] Run post-publication installation and CLI smoke tests from PyPI.

The licensing blocker is resolved: the package declares the SPDX expression
`Apache-2.0`, and both distribution formats carry the license and project notice.
No TestPyPI or PyPI publication is performed by ordinary CI or by the
implementation work described here.
