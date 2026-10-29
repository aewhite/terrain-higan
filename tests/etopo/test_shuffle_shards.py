import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from terrain_higan.etopo import shuffle_shards


class ShuffleShardsTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.dataset_root = Path(temporary_directory.name)
        for level in (0, 1, 2, 3, 4, 5, 10):
            for patch_size in (2, 3):
                directory = self.dataset_root / f"z{level}" / f"{patch_size}x{patch_size}"
                directory.mkdir(parents=True)
                np.full((1, patch_size, patch_size), level, dtype=np.float32).tofile(
                    directory / "shard-000000"
                )


    def test_only_selected_directories_are_shuffled(self):
        cases = [
            ([], {0, 1, 2, 3, 4, 5, 10}),
            (["--z-levels", "0", "1", "2", "3", "4"], {0, 1, 2, 3, 4}),
            (["--z-levels", "1", "4", "1"], {1, 4}),
        ]
        for arguments, expected_levels in cases:
            output = io.StringIO()
            with (
                self.subTest(arguments=arguments),
                redirect_stdout(output),
                patch.object(shuffle_shards, "shuffle_directory", return_value=(1, 1)) as shuffle,
            ):
                shuffle_shards.main([
                    "--base-dir", str(self.dataset_root), "--seed", "42", *arguments,
                ])

                expected_directories = {
                    self.dataset_root / f"z{level}" / f"{patch_size}x{patch_size}"
                    for level in expected_levels
                    for patch_size in (2, 3)
                }
                self.assertEqual(
                    {Path(call.args[0]) for call in shuffle.call_args_list}, expected_directories,
                )
                self.assertEqual(shuffle.call_count, len(expected_directories))
                self.assertTrue(all(call.args[2:] == (False, 42) for call in shuffle.call_args_list))
                self.assertIn(f"in {len(expected_directories)} directories", output.getvalue())


    def test_targeted_dry_run_leaves_all_files_unchanged(self):
        before = {path: path.read_bytes() for path in self.dataset_root.rglob("shard-*")}
        output = io.StringIO()

        with redirect_stdout(output):
            shuffle_shards.main([
                "--base-dir", str(self.dataset_root), "--z-levels", "1", "--dry-run",
            ])

        self.assertIn(str(self.dataset_root / "z1" / "2x2"), output.getvalue())
        self.assertIn(str(self.dataset_root / "z1" / "3x3"), output.getvalue())
        self.assertNotIn(str(self.dataset_root / "z10"), output.getvalue())
        self.assertIn("Would process 2 directories", output.getvalue())
        self.assertEqual(
            {path: path.read_bytes() for path in self.dataset_root.rglob("shard-*")}, before,
        )
        self.assertFalse(list(self.dataset_root.rglob("*.tmp")))
        self.assertFalse(list(self.dataset_root.rglob("*.old")))


    def test_unmatched_levels_fail_without_shuffling(self):
        errors = io.StringIO()
        with (
            redirect_stderr(errors),
            patch.object(shuffle_shards, "shuffle_directory") as shuffle,
            self.assertRaises(SystemExit) as error,
        ):
            shuffle_shards.main(["--base-dir", str(self.dataset_root), "--z-levels", "99"])

        self.assertEqual(error.exception.code, 2)
        self.assertIn("No dataset directories found for the requested --z-levels", errors.getvalue())
        shuffle.assert_not_called()
