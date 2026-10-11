#!/usr/bin/env python3
"""CPU-only checks for release preflight and workflow publication guards."""

import importlib.util
import io
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "release_preflight", ROOT / "scripts" / "release_preflight.py"
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)

PYPROJECT = """[project]
name = "matmul-inspector"
version = "0.1.0"
license = "Apache-2.0"
"""


def git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()


def metadata(version="0.1.0", extra=""):
    return (
        "Metadata-Version: 2.4\nName: matmul-inspector\n"
        f"Version: {version}\nLicense-Expression: Apache-2.0\n{extra}\n"
    ).encode()


def tar_file(archive, name, data):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))


def distributions(directory, version="0.1.0", extra=""):
    wheel = directory / "matmul_inspector-0.1.0-py3-none-any.whl"
    core = metadata(version, extra)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("matmul_inspector-0.1.0.dist-info/METADATA", core)
        archive.writestr("matmul_inspector-0.1.0.dist-info/licenses/LICENSE", "license")
        archive.writestr("matmul_inspector-0.1.0.dist-info/licenses/NOTICE", "notice")
    sdist = directory / "matmul_inspector-0.1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        tar_file(archive, "matmul_inspector-0.1.0/PKG-INFO", core)
        tar_file(archive, "matmul_inspector-0.1.0/LICENSE", b"license")
        tar_file(archive, "matmul_inspector-0.1.0/NOTICE", b"notice")
    return wheel, sdist


class ReleasePreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="matmul-release-test-")
        self.root = Path(self.temp.name)
        (self.root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
        git(self.root, "init", "-b", "main")
        git(self.root, "config", "user.name", "Release Test")
        git(self.root, "config", "user.email", "release@example.invalid")
        git(self.root, "add", "pyproject.toml")
        git(self.root, "commit", "-m", "initial")
        self.main_commit = git(self.root, "rev-parse", "HEAD")
        git(self.root, "tag", "v0.1.0")

    def tearDown(self):
        self.temp.cleanup()

    def preflight(self, event="push", ref="refs/tags/v0.1.0", sha=None):
        return release.release_preflight(
            event_name=event, ref=ref, sha=sha or self.main_commit,
            root=self.root, main_ref="main",
        )

    def test_valid_tag_push_requires_exact_commit_on_main(self):
        result = self.preflight()
        self.assertEqual(result["publish_allowed"], "true")
        self.assertEqual(result["tagged_commit"], self.main_commit)

    def test_manual_dispatch_is_always_build_only_even_on_release_tag(self):
        result = self.preflight(event="workflow_dispatch", sha="not-a-commit")
        self.assertEqual(result["publish_allowed"], "false")
        self.assertEqual(result["validation_mode"], "build-only")

    def test_invalid_tag_and_version_mismatch_fail_closed(self):
        for ref in ("refs/heads/main", "refs/tags/v0.1", "refs/tags/v0.1.0-rc1",
                    "refs/tags/v00.1.0"):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                self.preflight(ref=ref)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.preflight(ref="refs/tags/v0.2.0")

    def test_non_main_tagged_commit_fails_ancestry(self):
        git(self.root, "switch", "-c", "feature")
        (self.root / "feature").write_text("not on main\n", encoding="utf-8")
        git(self.root, "add", "feature")
        git(self.root, "commit", "-m", "feature")
        feature_commit = git(self.root, "rev-parse", "HEAD")
        git(self.root, "tag", "-f", "v0.1.0")
        with self.assertRaisesRegex(ValueError, "not an ancestor"):
            self.preflight(sha=feature_commit)

    def test_tag_and_workflow_commit_must_match(self):
        git(self.root, "commit", "--allow-empty", "-m", "later")
        with self.assertRaisesRegex(ValueError, "not workflow commit"):
            self.preflight(sha=git(self.root, "rev-parse", "HEAD"))


class DistributionValidationTests(unittest.TestCase):
    def test_valid_pair_records_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            distributions(Path(directory))
            records = release.validate_artifacts(Path(directory), "0.1.0")
            self.assertEqual(len(records), 2)
            self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in records))

    def test_unexpected_and_missing_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel, sdist = distributions(root)
            (root / "user-supplied.zip").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                release.validate_artifacts(root, "0.1.0")
            (root / "user-supplied.zip").unlink()
            sdist.unlink()
            with self.assertRaises(ValueError):
                release.validate_artifacts(root, "0.1.0")
            self.assertTrue(wheel.exists())

    def test_distribution_filenames_must_match_name_and_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel, _ = distributions(root)
            wheel.rename(root / "different-0.1.0-py3-none-any.whl")
            with self.assertRaisesRegex(ValueError, "filenames do not match"):
                release.validate_artifacts(root, "0.1.0")

    def test_version_and_conflicting_metadata_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            distributions(Path(directory), version="0.2.0")
            with self.assertRaisesRegex(ValueError, "does not match"):
                release.validate_artifacts(Path(directory), "0.1.0")
        with tempfile.TemporaryDirectory() as directory:
            distributions(Path(directory), extra="Version: 0.1.0\n")
            with self.assertRaisesRegex(ValueError, "conflicting Version"):
                release.validate_artifacts(Path(directory), "0.1.0")


class WorkflowConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        cls.trigger = cls.workflow.split("\nconcurrency:", 1)[0]
        cls.build, cls.publish = cls.workflow.split("\n  publish:", 1)

    def test_only_tag_push_and_manual_dispatch_trigger(self):
        self.assertIn("tags:", self.trigger)
        self.assertIn('"v*"', self.trigger)
        self.assertIn("workflow_dispatch:", self.trigger)
        self.assertNotIn("pull_request:", self.trigger)
        self.assertNotIn("branches:", self.trigger)

    def test_publish_guards_and_environment(self):
        for guard in ("publish_allowed == 'true'", "github.event_name == 'push'",
                      "github.ref_type == 'tag'",
                      "needs.build-and-verify.result == 'success'"):
            self.assertIn(guard, self.publish)
        self.assertRegex(self.publish, r"environment:\s+name: pypi")

    def test_oidc_only_in_publish_and_no_permanent_credential(self):
        self.assertNotIn("id-token: write", self.build)
        self.assertEqual(self.publish.count("id-token: write"), 1)
        self.assertNotRegex(self.workflow.lower(), r"password:|api-token:|skip-existing")

    def test_publish_reuses_exact_build_artifact_without_checkout(self):
        self.assertEqual(self.workflow.count("python -m build"), 1)
        self.assertIn("artifact_id: ${{ steps.upload.outputs.artifact-id }}", self.build)
        self.assertIn(
            "artifact-ids: ${{ needs.build-and-verify.outputs.artifact_id }}", self.publish
        )
        self.assertNotIn("actions/checkout@", self.publish)

    def test_actions_are_pinned_to_commit_shas(self):
        uses = re.findall(r"^\s*uses:\s*[^@\s]+@([^\s]+)", self.workflow, re.MULTILINE)
        self.assertTrue(uses)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{40}", value) for value in uses))


if __name__ == "__main__":
    unittest.main()
