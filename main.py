"""
Entry point for Brain Typing.

Runs the CNN-RNN-AE-XGBoost pipeline defined in src/model.py.
Execute from the repository root:

    python main.py

Ensure the data file path inside src/model.py points to your local .mat file
before running. See data/README.md for dataset details.
"""

import runpy

runpy.run_module("src.model", run_name="__main__", alter_sys=True)
