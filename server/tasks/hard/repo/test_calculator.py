# test_calculator.py — unittest-based test suite
# This file MUST be migrated from unittest to pytest.
# Has: setUp, tearDown, setUpClass, tearDownClass, assertRaises,
#      assertAlmostEqual, assertIn, @skip, @expectedFailure

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from calculator import Calculator


class TestCalculatorBasicOps(unittest.TestCase):
    """Test basic arithmetic operations."""

    def setUp(self):
        """Create a fresh Calculator for each test."""
        self.calc = Calculator()

    def tearDown(self):
        """Clean up after each test."""
        self.calc = None

    def test_add(self):
        self.assertEqual(self.calc.add(2, 3), 5)

    def test_add_negative(self):
        self.assertEqual(self.calc.add(-1, -1), -2)

    def test_add_floats(self):
        self.assertAlmostEqual(self.calc.add(0.1, 0.2), 0.3, places=10)

    def test_subtract(self):
        self.assertEqual(self.calc.subtract(10, 4), 6)

    def test_subtract_negative_result(self):
        self.assertEqual(self.calc.subtract(3, 7), -4)

    def test_multiply(self):
        self.assertEqual(self.calc.multiply(3, 4), 12)

    def test_multiply_by_zero(self):
        self.assertEqual(self.calc.multiply(5, 0), 0)

    def test_divide(self):
        self.assertEqual(self.calc.divide(10, 2), 5.0)

    def test_divide_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            self.calc.divide(1, 0)

    def test_divide_by_zero_message(self):
        with self.assertRaises(ZeroDivisionError) as ctx:
            self.calc.divide(10, 0)
        self.assertIn("Cannot divide by zero", str(ctx.exception))


class TestCalculatorMemory(unittest.TestCase):
    """Test memory store/recall with class-level setup."""

    shared_calc = None

    @classmethod
    def setUpClass(cls):
        """Create a shared calculator instance for the class."""
        cls.shared_calc = Calculator()
        cls.shared_calc.store(99.9)

    @classmethod
    def tearDownClass(cls):
        """Clean up the shared instance."""
        cls.shared_calc = None

    def setUp(self):
        self.calc = Calculator()

    def test_initial_memory_is_zero(self):
        self.assertEqual(self.calc.recall(), 0.0)

    def test_store_and_recall(self):
        self.calc.store(42.5)
        self.assertEqual(self.calc.recall(), 42.5)

    def test_store_overwrites(self):
        self.calc.store(10)
        self.calc.store(20)
        self.assertEqual(self.calc.recall(), 20)

    def test_shared_instance_has_stored_value(self):
        """Verify setUpClass stored a value in the shared instance."""
        self.assertEqual(self.shared_calc.recall(), 99.9)


class TestCalculatorHistory(unittest.TestCase):
    """Test calculation history tracking."""

    def setUp(self):
        self.calc = Calculator()

    def test_history_starts_empty(self):
        self.assertEqual(self.calc.get_history(), [])

    def test_history_records_operations(self):
        self.calc.add(1, 2)
        self.calc.multiply(3, 4)
        history = self.calc.get_history()
        self.assertEqual(len(history), 2)
        self.assertIn("1 + 2 = 3", history[0])
        self.assertIn("3 * 4 = 12", history[1])

    def test_clear_history(self):
        self.calc.add(1, 1)
        self.calc.clear_history()
        self.assertEqual(self.calc.get_history(), [])

    def test_history_after_all_ops(self):
        self.calc.add(1, 2)
        self.calc.subtract(5, 3)
        self.calc.multiply(2, 3)
        self.calc.divide(10, 2)
        self.assertEqual(len(self.calc.get_history()), 4)

    @unittest.skip("Not implemented yet — negative history indexing")
    def test_history_negative_index(self):
        self.calc.add(1, 2)
        self.calc.add(3, 4)
        pass

    @unittest.expectedFailure
    def test_history_returns_copy(self):
        """History should return a copy, mutating it shouldn't affect internal state.
        This is marked as expected failure for demonstration."""
        self.calc.add(1, 2)
        history = self.calc.get_history()
        history.clear()
        # This actually passes because get_history returns list(self.history),
        # but we mark it expectedFailure for migration testing purposes.
        self.assertNotEqual(self.calc.get_history(), [])


if __name__ == "__main__":
    unittest.main()
