"""Task B in-silico-perturbation runners, one module per competitor model.

A model's module is imported only once it has been selected on the command line. GEARS
and scGPT run under their own virtualenvs (cache/.gears_venv, cache/.scgpt_venv) whose
dependency sets cannot be imported alongside the KOT stack, so importing all five in one
process is not possible and the registry stays a table of module paths.
"""
from __future__ import annotations

MODELS = {
    "gears": "src.evaluation.isp.gears",
    "linear": "src.evaluation.isp.linear",
    "multipert": "src.evaluation.isp.multipert",
    "regvelo": "src.evaluation.isp.regvelo",
    "scgpt": "src.evaluation.isp.scgpt",
}
