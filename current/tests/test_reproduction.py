import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_release
import run
import summarize


class PublicationGuardTests(unittest.TestCase):
    def test_editorial_and_training_files_are_rejected(self):
        for name in ("main.pdf", "supplement.tex", "papers/article.md", "cover_letter.docx",
                     "source.zip", "cas-sc.cls", "model.pt", "data.npz", ".env"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                check_release.check_file(name, b"test")

    def test_disguised_documents_are_rejected(self):
        for data in (b"%PDF-1.5", b"PK\x03\x04", b"\\" + b"documentclass{article}"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                check_release.check_file("current/readme.txt", data)

    def test_credentials_and_private_paths_are_rejected(self):
        samples = [b"ghp_" + b"a" * 36, b"AKIA" + b"A" * 16,
                   b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----", b"/home/" + b"private/project/"]
        for data in samples:
            with self.subTest(data=data), self.assertRaises(ValueError):
                check_release.check_file("current/readme.txt", data)

    def test_scientific_code_is_allowed(self):
        check_release.check_file("current/model.py", b"def fit(x): return x\n")

    def test_only_current_release_layout_is_allowed(self):
        for name in ("src/model.py", "results/old.json", "configs/old.toml",
                     "experiment_snapshots/model.py", "SHA256SUMS.txt"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                check_release.check_file(name, b"test")
        for name in check_release.ROOT_FILES:
            with self.subTest(name=name):
                check_release.check_file(name, b"test")


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configurations = json.loads((run.ROOT / "protocols/runs.json").read_text())

    def arguments(self, row):
        return SimpleNamespace(**{k: row[k] for k in ("study", "design", "seed", "regime", "method")},
            output=Path("/tmp/reproduction-test"), data_home=Path("/tmp/reproduction-data"))

    def test_source_snapshot_matches(self):
        run.verify_source()

    def test_all_ordinary_runs_have_unique_output_names(self):
        destinations = set()
        for row in self.configurations:
            _, destination = run.command(self.arguments(row))
            self.assertNotIn(destination, destinations)
            destinations.add(destination)
        self.assertEqual(len(destinations), 3180)

    def test_shared_checkpoint_and_scientific_options(self):
        matched = [c for c in self.configurations if c["study"] == "image"
                   and c["design"] == "stl10_resnet50" and c["seed"] == 16200
                   and c["method"] in (*summarize.BASELINES, "ghvade_adaptive")]
        checkpoints = set()
        for row in matched:
            cmd, _ = run.command(self.arguments(row))
            checkpoints.add(cmd[cmd.index("--mixture-pretrain-checkpoint")+1])
            self.assertEqual(cmd[cmd.index("--mixture-cycles")+1], str(row["arguments"]["mixture_cycles"]))
            self.assertEqual(cmd[cmd.index("--latent-dim")+1], "8")
            self.assertEqual(cmd[cmd.index("--shape-score-margin")+1], "0.01")
        self.assertEqual(len(matched), 6)
        self.assertEqual(len(checkpoints), 1)

    def test_unknown_seed_cannot_be_run(self):
        args = self.arguments(self.configurations[0])
        args.seed = -1
        with self.assertRaises(ValueError):
            run.command(args)

    def test_holm_matches_known_example(self):
        rows = [dict(p=p) for p in (.01, .04, .03)]
        summarize.holm(rows)
        self.assertEqual([r["holm_p"] for r in rows], [.03, .06, .06])

    def test_contrast_edge_cases(self):
        self.assertEqual(summarize.contrast([0.] * 20)["p"], 1.)
        self.assertEqual(summarize.contrast([1.] * 20)["p"], 0.)
        for values in ([0.] * 19, [float("nan")] * 20):
            with self.assertRaises(ValueError):
                summarize.contrast(values)

    def test_full_statistical_reproduction(self):
        with redirect_stdout(io.StringIO()):
            summarize.main()
        report = json.loads((run.ROOT / "generated/summary.json").read_text())
        self.assertEqual(len(report["image_primary"]), 6)
        self.assertEqual(len(report["image_secondary"]), 18)
        self.assertEqual(len(report["simulation_cellwise"]), 20)
        self.assertEqual(len(report["image_mechanisms"]), 6)
        self.assertEqual(len(report["known_latent_average"]), 3)


if __name__ == "__main__":
    unittest.main()
