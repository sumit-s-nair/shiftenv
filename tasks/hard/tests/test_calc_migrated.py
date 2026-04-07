# test_calc_migrated.py — META-GRADER for the hard task (unittest → pytest)
# Validates that test_calculator.py was correctly migrated from unittest to pytest.
# Runs the migrated file via SUBPROCESS to avoid pytest collection conflicts.

import ast
import os
import subprocess
import sys
import pytest


REPO_DIR = os.path.join(os.path.dirname(__file__), "..", "repo")
MIGRATED_FILE = os.path.join(REPO_DIR, "test_calculator.py")


def _get_source() -> str:
    with open(MIGRATED_FILE, "r") as f:
        return f.read()


def _get_imports(filepath: str) -> set[str]:
    with open(filepath, "r") as f:
        tree = ast.parse(f.read(), filename=filepath)
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def _get_ast_tree():
    with open(MIGRATED_FILE, "r") as f:
        return ast.parse(f.read(), filename=MIGRATED_FILE)


# ===========================================================================
# 1. Import cleanliness
# ===========================================================================
class TestImportCleanliness:
    def test_no_unittest_import(self):
        imports = _get_imports(MIGRATED_FILE)
        assert "unittest" not in imports, "Still imports unittest"

    def test_pytest_imported(self):
        imports = _get_imports(MIGRATED_FILE)
        assert "pytest" in imports, "Does not import pytest"


# ===========================================================================
# 2. Structure: no TestCase classes, no self.assertX, proper fixtures
# ===========================================================================
class TestStructure:
    def test_no_testcase_classes(self):
        """No classes inheriting unittest.TestCase."""
        tree = _get_ast_tree()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = ""
                    if isinstance(base, ast.Attribute):
                        base_name = f"{getattr(base.value, 'id', '')}.{base.attr}"
                    elif isinstance(base, ast.Name):
                        base_name = base.id
                    assert "TestCase" not in base_name, (
                        f"Class {node.name} still inherits from TestCase"
                    )

    def test_no_self_assert_methods(self):
        """No self.assertEqual, self.assertRaises etc."""
        source = _get_source()
        bad = [m for m in [
            "self.assertEqual", "self.assertNotEqual", "self.assertTrue",
            "self.assertFalse", "self.assertIn", "self.assertNotIn",
            "self.assertAlmostEqual", "self.assertRaises",
            "self.assertIsNone", "self.assertIsNotNone",
        ] if m in source]
        assert not bad, f"Found unittest assert methods: {bad}"

    def test_uses_pytest_raises(self):
        """Must use pytest.raises for exception testing."""
        source = _get_source()
        assert "pytest.raises" in source, "Should use pytest.raises(ZeroDivisionError)"

    def test_no_setup_teardown(self):
        """setUp/tearDown should be replaced with fixtures."""
        source = _get_source()
        assert "def setUp" not in source, "Still has setUp method"
        assert "def tearDown" not in source, "Still has tearDown method"
        assert "def setUpClass" not in source, "Still has setUpClass method"
        assert "def tearDownClass" not in source, "Still has tearDownClass method"

    def test_uses_pytest_fixture(self):
        """Must use @pytest.fixture for setup."""
        source = _get_source()
        assert "pytest.fixture" in source or "@pytest.fixture" in source, (
            "Should use @pytest.fixture for test setup"
        )

    def test_uses_pytest_mark_skip(self):
        """@unittest.skip → @pytest.mark.skip."""
        source = _get_source()
        assert "unittest.skip" not in source, "Still uses @unittest.skip"
        assert "pytest.mark.skip" in source, "Should use @pytest.mark.skip"

    def test_no_unittest_expected_failure(self):
        """@unittest.expectedFailure → @pytest.mark.xfail."""
        source = _get_source()
        assert "unittest.expectedFailure" not in source, "Still uses @unittest.expectedFailure"
        assert "pytest.mark.xfail" in source, "Should use @pytest.mark.xfail"

    def test_no_unittest_main(self):
        """Should not have if __name__ == '__main__': unittest.main()."""
        source = _get_source()
        assert "unittest.main" not in source, "Still has unittest.main()"

    def test_has_enough_test_functions(self):
        """Should have at least 15 top-level test functions."""
        tree = _get_ast_tree()
        test_funcs = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        ]
        assert len(test_funcs) >= 15, (
            f"Only {len(test_funcs)} test functions, expected at least 15"
        )


# ===========================================================================
# 3. Execution — run the migrated file with pytest via subprocess
# ===========================================================================
class TestExecution:
    def test_migrated_tests_pass(self):
        """Run the migrated test file and verify it passes."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", MIGRATED_FILE, "--tb=short", "-q"],
            capture_output=True, text=True, timeout=30, cwd=REPO_DIR,
        )
        assert result.returncode == 0, (
            f"Migrated tests failed!\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_migrated_test_count(self):
        """Verify a reasonable number of tests are collected."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", MIGRATED_FILE, "--tb=no", "-q", "--co"],
            capture_output=True, text=True, timeout=30, cwd=REPO_DIR,
        )
        lines = [l for l in result.stdout.strip().splitlines() if "::" in l]
        assert len(lines) >= 15, (
            f"Expected at least 15 collected tests, got {len(lines)}"
        )
