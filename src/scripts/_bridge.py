import importlib.util
from pathlib import Path


def load_module(name: str):
    source_path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_repo_{name}", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load project module {name!r} from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
