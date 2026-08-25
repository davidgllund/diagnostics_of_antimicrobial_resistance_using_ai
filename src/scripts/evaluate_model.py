from scripts._bridge import load_module

_module = load_module("evaluate_model")
for _name in dir(_module):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_module, _name)
