import os
import sys
from pathlib import Path

import pytest

# Get the root directory of the project
root_dir = Path(__file__).parent.parent

# Get the list of all Python files in the examples directory
example_files = list(root_dir.glob("examples/**/*.py"))

# Convert the file paths to module names
modules_to_test: list[str] = [
    str(p.relative_to(root_dir)).replace(os.sep, ".").rstrip(".py")
    for p in example_files
]

@pytest.mark.parametrize("module_name", modules_to_test)
def test_examples(module_name):
    __import__(module_name)