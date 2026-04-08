"""Validator-friendly grader entry points.

These wrappers mirror the successful sample-repo pattern by exposing stable,
top-level grader callables that validators can import from repo root.
"""

from tasks.easy.tests.grader import grader as _easy
from tasks.medium.tests.grader import grader as _medium
from tasks.hard.tests.grader import grader as _hard


def easy_grader(trajectory: dict | None = None) -> float:
    return _easy(trajectory=trajectory)


def medium_grader(trajectory: dict | None = None) -> float:
    return _medium(trajectory=trajectory)


def hard_grader(trajectory: dict | None = None) -> float:
    return _hard(trajectory=trajectory)
