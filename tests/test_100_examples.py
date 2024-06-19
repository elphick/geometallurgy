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

# Check if we are in a CI/CD environment
on_ci_cd = os.environ.get('ON_CI_CD', 'false').lower() == 'true'


@pytest.mark.parametrize("module_name", modules_to_test)
@pytest.mark.skipif(on_ci_cd, reason="Skip on CI/CD")
def test_examples(module_name):
    os.environ["PYVISTA_OFF_SCREEN"] = "True"
    __import__(module_name)