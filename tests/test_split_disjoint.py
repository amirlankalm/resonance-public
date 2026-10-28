import hashlib
import os
from pathlib import Path
import unittest

from resonance.splits import (
    DEV_SEEDS,
    TEST_SALT_COMMITMENT_SHA256,
    assert_disjoint_splits,
    derive_test_seeds,
    get_dev_seeds,
    get_train_seeds,
)


class TestSplitDisjoint(unittest.TestCase):
    def test_train_and_dev_disjointness(self):
        dev = get_dev_seeds()
        self.assertEqual(len(dev), 50)
        self.assertEqual(dev[0], 10000)
        self.assertEqual(dev[-1], 10049)

        # SFT training seeds
        sft_train = get_train_seeds(count=500, start_seed=50000)
        assert_disjoint_splits(sft_train, dev)

        # RL training seeds
        rl_train = list(range(30000, 31000))
        assert_disjoint_splits(rl_train, dev)

    def test_test_salt_commitment_validation(self):
        """Verify that unauthorized salt strings are rejected by commitment check."""
        with self.assertRaises(ValueError):
            derive_test_seeds("unauthorized_guess_string", count=10)

    def test_mock_salt_derivation_and_disjointness(self):
        """Verify derivation logic and mathematical disjointness using ephemeral test salt."""
        # Ephemeral mock salt for testing algorithm without exposing production salt
        mock_salt = "ephemeral_mock_salt_for_testing_disjointness_only"
        mock_commitment = hashlib.sha256(mock_salt.encode("utf-8")).hexdigest()

        test_seeds = derive_test_seeds(mock_salt, count=100, expected_commitment=mock_commitment)
        self.assertEqual(len(test_seeds), 100)
        self.assertTrue(all(s >= 100000 for s in test_seeds))

        dev_seeds = get_dev_seeds()
        train_seeds = list(range(30000, 99999))

        assert_disjoint_splits(train_seeds, dev_seeds, test_seeds)

    def test_local_production_salt_if_present(self):
        """If production salt exists in secure local storage, verify its commitment and disjointness."""
        salt_file = Path(os.path.expanduser("~/.resonance_heldout_salt"))
        if salt_file.is_file():
            prod_salt = salt_file.read_text(encoding="utf-8").strip()
            test_seeds = derive_test_seeds(prod_salt, count=100)
            self.assertEqual(len(test_seeds), 100)
            self.assertTrue(all(s >= 100000 for s in test_seeds))
            assert_disjoint_splits(list(range(30000, 99999)), get_dev_seeds(), test_seeds)

    def test_new_salt_never_committed_in_git(self):
        """Verify that the production salt was never committed anywhere in git history or code."""
        import subprocess

        salt_file = Path(os.path.expanduser("~/.resonance_heldout_salt"))
        if salt_file.is_file():
            prod_salt = salt_file.read_text(encoding="utf-8").strip()
            if prod_salt:
                # 1. Verify not in current working tree or index
                res_grep = subprocess.run(
                    ["git", "grep", prod_salt],
                    cwd=str(Path(__file__).resolve().parent.parent),
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(res_grep.returncode, 0, "Production salt must NOT appear in tracked files")

                # 2. Verify never in any commit history
                res_log = subprocess.run(
                    ["git", "log", "-S", prod_salt, "--oneline"],
                    cwd=str(Path(__file__).resolve().parent.parent),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    res_log.stdout.strip(),
                    "",
                    "Production salt was committed in git history! Zero-tolerance violation.",
                )


if __name__ == "__main__":
    unittest.main()
