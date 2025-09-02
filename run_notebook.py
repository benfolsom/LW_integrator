import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

nb_path = r"c:\Users\benfol\Documents\LW_windows\LW_integrator\synchrotron_calc.ipynb"
out_path = r"c:\Users\benfol\Documents\LW_windows\LW_integrator\synchrotron_calc_executed.ipynb"

print("Reading", nb_path)
nb = nbformat.read(nb_path, as_version=4)

# Use the kernel that was registered for the notebook (.venv)
ep = ExecutePreprocessor(timeout=600, kernel_name=".venv (Python 3.11.9)")
try:
    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(nb_path)}})
except Exception as e:
    print("Execution failed:", e)
    raise

nbformat.write(nb, out_path)
print("Executed notebook saved to", out_path)
