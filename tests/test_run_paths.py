"""Verify checkpoint selection for every archived Ray module without Ray imports."""
import ast
import subprocess
import sys
import runpy
import tempfile
import unittest
from pathlib import Path


class RunPathsTests(unittest.TestCase):
    def test_local_paths_and_selection(self):
        root = Path(__file__).resolve().parents[1]
        sources = list(root.glob('eco_evolutionary*/run_paths.py'))
        self.assertEqual(len(sources), 9)
        for source in sources:
            with self.subTest(module=source.parent.name), tempfile.TemporaryDirectory() as tmp:
                namespace = runpy.run_path(str(source))
                self.assertEqual(namespace['RAY_RESULTS_DIR'], source.parent / 'runs')
                latest = namespace['latest_checkpoint']
                runs = Path(tmp)
                latest.__globals__['RAY_RESULTS_DIR'] = runs
                with self.assertRaises(FileNotFoundError):
                    latest()
                def checkpoint(experiment, number, complete=True):
                    directory = runs / experiment / 'trial' / f'checkpoint_{number:06d}'
                    directory.mkdir(parents=True)
                    (directory / 'rllib_checkpoint.json').write_text('{}')
                    if complete:
                        for policy in namespace['POLICY_IDS']:
                            policy_dir = directory / 'learner_group/learner/rl_module' / policy
                            policy_dir.mkdir(parents=True)
                            for name in ('metadata.json', 'class_and_ctor_args.pkl', 'module_state.pkl'):
                                (policy_dir / name).write_text('{}')
                    return directory
                checkpoint('PPO_TEST_2026-07-01_10-00-00', 99)
                expected = checkpoint('PPO_TEST_2026-07-02_10-00-00', 2)
                checkpoint('PPO_TEST_2026-07-02_10-00-00', 1)
                checkpoint('PPO_TEST_2026-07-02_10-00-00', 3, complete=False)
                checkpoint('PPO_TEST_2026-07-03_10-00-00', 4, complete=False)
                for variant in ('NEUTRAL_CONTROL', 'SCALED', 'FIXED025'):
                    checkpoint(f'PPO_TEST_{variant}_2026-07-04_10-00-00', 100)
                self.assertEqual(latest(), expected)

    def test_consumer_imports_in_direct_script_context(self):
        root = Path(__file__).resolve().parents[1]
        checked = 0
        with tempfile.TemporaryDirectory() as tmp:
            for module in root.glob('eco_evolutionary*'):
                for source in module.glob('*.py'):
                    for node in ast.parse(source.read_text()).body:
                        if isinstance(node, ast.Try) and any(
                            isinstance(child, ast.ImportFrom) and
                            (child.module or '').endswith('.run_paths')
                            for child in node.body
                        ):
                            code = ('import sys; sys.dont_write_bytecode=True; '
                                    f'sys.path.insert(0, {str(module)!r})\n' + ast.unparse(node))
                            result = subprocess.run([sys.executable, '-I', '-c', code],
                                                    cwd=tmp, capture_output=True, text=True)
                            self.assertEqual(result.returncode, 0, f'{source}: {result.stderr}')
                            checked += 1
        self.assertEqual(checked, 57)


if __name__ == '__main__':
    unittest.main()
