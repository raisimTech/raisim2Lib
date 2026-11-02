"""CTest launcher: use the just-built extension and its matching runtime libraries."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--module-dir", required=True)
parser.add_argument("--library-dir", required=True)
args = parser.parse_args()
env = os.environ.copy()
for variable, directory in [("PYTHONPATH", args.module_dir),
                            ("PATH" if os.name == "nt" else "DYLD_LIBRARY_PATH" if sys.platform == "darwin"
                             else "LD_LIBRARY_PATH", args.library_dir)]:
    env[variable] = directory + os.pathsep + env.get(variable, "")
result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                         str(Path(__file__).resolve().parent), "-v"], env=env)
sys.exit(result.returncode)
