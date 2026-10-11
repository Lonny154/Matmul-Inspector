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

## Production release workflow

`.github/workflows/release.yml` separates release construction from publication.
The build job has read-only repository permission and creates the wheel and source
distribution exactly once. It validates their contents and metadata, installs each
archive in a clean environment, runs the installed-package smoke suite, runs
`twine check --strict`, records SHA-256 hashes in the job summary, and uploads
only the validated `.whl` and `.tar.gz` files as a workflow artifact.

The publish job never rebuilds or accepts an externally supplied archive. It
downloads the build job's artifact by the artifact ID emitted in the same workflow
run. Only this job has `id-token: write`, and it enters the protected GitHub
environment named `pypi` before invoking PyPA's publishing action. No PyPI token
or password is stored in the repository.

Normal production publication requires all of the following:

1. The event is a tag push, not a pull request, branch push, or manual dispatch.
2. The ref exactly matches `vMAJOR.MINOR.PATCH` with no prerelease suffix or
   leading-zero component.
3. The tag version equals `project.version` in `pyproject.toml`.
4. The tag ref and workflow SHA resolve to the same commit.
5. `git merge-base --is-ancestor` confirms that exact commit is in the freshly
   fetched `origin/main` history.
6. Every build, audit, metadata, installation, and Twine check succeeds.

The workflow's concurrency group prevents overlapping runs for the same ref.
Failures stop before artifact upload or publishing; publication does not use
`skip-existing`.

### Manual build-only validation

Before approving the first release, open **Actions → PyPI release → Run workflow**
and select the `build-only` mode on the branch or commit to validate. A normal
manual run is labeled
**BUILD-ONLY**. It performs the complete build and verification pipeline and
uploads inspectable distributions, but its preflight always emits
`publish_allowed=false`. This remains true even if the manual run is invoked
against a tag. The publish job is skipped, the `pypi` environment is not entered,
and no OIDC publishing token is requested. The sole exception is the narrowly
fixed v0.1.0 recovery mode described below; no user-supplied ref, version, SHA,
artifact ID, or URL is accepted by that mode.

Review the build job summary for the two filenames, byte sizes, and SHA-256 hashes.
Download the `validated-release-distributions` artifact and independently check:

```sh
sha256sum matmul_inspector-0.1.0*
python -m twine check --strict matmul_inspector-0.1.0*
```

The logs also show the archive audit, embedded project name and version,
`Apache-2.0` license expression, LICENSE/NOTICE presence, and clean-install smoke
results. Workflow artifacts expire after 14 days and are not PyPI releases.

### Approved production publication

After the exact commit on `main` and its build-only artifacts have been reviewed,
create and push the approved version tag:

```sh
git switch main
git pull --ff-only origin main
git tag -a v0.1.0 -m "Matmul Inspector v0.1.0"
git push origin v0.1.0
```

Do not use this procedure until tag creation and publication are explicitly
approved. The tag push starts a new workflow run that rebuilds from the tagged
source and repeats every validation. The GitHub `pypi` environment applies its
deployment tag restriction and any configured protection rules. GitHub then
requests a short-lived OIDC identity for the publish job. PyPI's pending trusted
publisher matches the repository, `release.yml` workflow, and `pypi` environment;
on the first successful publication it creates the `matmul-inspector` project
and records that release.

Production PyPI is intentionally separate from any future TestPyPI workflow.
TestPyPI configuration must not be assumed and is not used as a fallback.

### Temporary v0.1.0 recovery mode

The original v0.1.0 tag-triggered run validated its archives but failed during
publication because the downloaded files were nested below `dist/`. The tag must
not be moved or recreated. The temporary recovery choice
`recover-v0.1.0-from-c9d93c0` is therefore hard-coded to the existing annotated
tag object `6b7a2aeaeb62c1d6f8d8edf7641b1cffe0a36e7e` and its peeled source commit
`c9d93c0c7f9967b67e3da0776ccf0e67ee518e55`.

Before an approved recovery run, an administrator must temporarily allow the
protected `pypi` GitHub environment to deploy from the `main` branch in addition
to its release-tag rule. This is a manual repository-setting change; the workflow
does not change GitHub or PyPI configuration. Then dispatch the workflow from
the current `main` branch and select exactly `recover-v0.1.0-from-c9d93c0`.

The recovery preflight freshly fetches `origin/main` and the v0.1.0 tag, requires
the workflow controller SHA to equal `origin/main`, verifies both fixed tag
identities and ancestry, reads version 0.1.0 metadata from the tagged commit,
and only then checks out that source commit in detached-HEAD mode. The workflow
checks `HEAD` again immediately before building. Publication has a separate
authorization branch for this exact manual event and still uses only the two
validated archives uploaded by the build job in the same run. The flat `dist/`
layout check, protected environment, and publish-job-only OIDC permission remain
mandatory. Every other manual input remains build-only or fails closed.

After a successful publication and clean installation verification:

1. Remove the temporary `main` deployment allowance from the `pypi` environment.
2. Remove the recovery input, constants, preflight branch, checkout step, publish
   authorization branch, recovery tests, and this subsection through normal PR
   review. Preserve the ordinary `build-only` manual workflow.
3. Do not move, delete, or recreate `v0.1.0`; retain its annotated tag object and
   peeled source commit as release provenance.

### Post-publication and failure handling

PyPI distribution files and version numbers are immutable: a filename cannot be
overwritten and version `0.1.0` cannot be reused for corrected artifacts. After
publication, create a clean environment outside the checkout and verify the exact
published version:

```sh
python3 -m venv /tmp/matmul-inspector-pypi
/tmp/matmul-inspector-pypi/bin/python -m pip install \
  --no-cache-dir "matmul-inspector==0.1.0"
/tmp/matmul-inspector-pypi/bin/matmul-inspector-python --version
```

If validation fails before publication, fix the source through normal review and
create a new tag only after the fix is on `main`. Never move or reuse a pushed
release tag. If PyPI accepts only part of a release, do not rerun with
`skip-existing` or attempt to replace files. Record what PyPI accepted, stop,
and prepare a new incremented package version after reviewing the failure. Create
a GitHub Release only after PyPI contents and clean installation are verified.

## v0.1.0 release checklist

- [x] Distribution name and version recorded as `matmul-inspector` 0.1.0.
- [x] Python 3.10–3.13 Linux build and clean-install CI defined.
- [x] Wheel and sdist content audit defined.
- [x] Runtime dependencies remain empty.
- [x] License the project under Apache License 2.0 and include `LICENSE` and
  `NOTICE` in both distribution formats.
- [ ] Confirm `matmul-inspector` availability through the PyPI publication workflow.
- [ ] Review distribution metadata and project URLs.
- [ ] Run and review the manual BUILD-ONLY release workflow.
- [ ] Optional future work: configure and validate a separate TestPyPI workflow.
- [ ] Publish the approved archives to PyPI.
- [ ] Create the approved `v0.1.0` Git tag and GitHub release.
- [ ] Run post-publication installation and CLI smoke tests from PyPI.

The licensing blocker is resolved: the package declares the SPDX expression
`Apache-2.0`, and both distribution formats carry the license and project notice.
No TestPyPI or PyPI publication is performed by ordinary CI or by the
implementation work described here.
