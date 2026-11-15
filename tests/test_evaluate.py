import math
import unittest

from evaluate import evaluate_loss


class EvaluationTests(unittest.TestCase):
    def test_empty_evaluation_loader_is_rejected(self):
        class Empty:
            def __iter__(self):
                return iter(())
        class Model:
            def eval(self):
                pass
        with self.assertRaises(ValueError):
            evaluate_loss(Model(), Empty(), "cpu")


if __name__ == "__main__":
    unittest.main()
