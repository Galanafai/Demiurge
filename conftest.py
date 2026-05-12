"""Root conftest.py: ensures src/ is on sys.path for all pytest invocations.

This allows tests to import modules as `from scene.schema import SceneTensor`
rather than `from src.scene.schema import SceneTensor`, consistent with the
editable install layout defined in pyproject.toml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
