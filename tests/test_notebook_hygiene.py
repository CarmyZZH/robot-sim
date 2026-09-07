import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


class NotebookHygieneTests(unittest.TestCase):
    def test_tracked_notebooks_are_clean_and_reviewable(self):
        notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
        self.assertTrue(notebooks, "Expected at least one tracked notebook")

        for path in notebooks:
            with self.subTest(notebook=path.name):
                document = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(document.get("nbformat"), 4)

                for cell_index, cell in enumerate(document.get("cells", [])):
                    self.assertIn(
                        "id", cell, f"{path.name} cell {cell_index} has no stable ID"
                    )
                    if cell.get("cell_type") != "code":
                        continue
                    self.assertIsNone(
                        cell.get("execution_count"),
                        f"{path.name} cell {cell_index} has an execution count",
                    )
                    self.assertEqual(
                        cell.get("outputs", []),
                        [],
                        f"{path.name} cell {cell_index} contains saved output",
                    )


if __name__ == "__main__":
    unittest.main()
