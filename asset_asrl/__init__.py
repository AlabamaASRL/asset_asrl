import os
import sys
from pathlib import Path
from importlib.metadata import distribution

# On Windows, the ASSET extension depends on Intel OpenMP.
# Locate the installed intel-openmp package and determine the environment root.
if sys.platform == "win32":
    dist = distribution("intel-openmp")
    env_root = Path(dist.locate_file("")).parent.parent

    # Intel OpenMP installs its DLLs under the environment's Library/bin directory.
    dll_dir = env_root / "Library" / "bin"

    # Add the DLL directory to the Windows DLL search path before importing ASSET.
    if dll_dir.is_dir():
        print(f"Found {dll_dir} ... adding to DLL search path")
        os.add_dll_directory(str(dll_dir))

# Import the compiled ASSET backend after its native dependencies are available.
import asset as _asset
import asset_asrl.VectorFunctions
import asset_asrl.OptimalControl
import asset_asrl.Utils
import asset_asrl.Astro
import asset_asrl.Solvers
import inspect

SoftwareInfo = _asset.SoftwareInfo

if __name__ == "__main__":
    _asset.SoftwareInfo()
    mlist = inspect.getmembers(_asset)
    for m in mlist:
        print(m[0], '= _asset.' + str(m[0]))