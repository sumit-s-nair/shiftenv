# calculator.py — A Calculator class (the code being tested)
# This file does NOT need migration. Only the test file does.


class Calculator:
    """A simple calculator with memory."""

    def __init__(self):
        self.memory = 0.0
        self.history: list[str] = []

    def add(self, a: float, b: float) -> float:
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def subtract(self, a: float, b: float) -> float:
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result

    def multiply(self, a: float, b: float) -> float:
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result

    def store(self, value: float) -> None:
        """Store a value in memory."""
        self.memory = value

    def recall(self) -> float:
        """Recall the stored memory value."""
        return self.memory

    def clear_history(self) -> None:
        """Clear the calculation history."""
        self.history.clear()

    def get_history(self) -> list[str]:
        """Return a copy of the calculation history."""
        return list(self.history)
