"""Rebuild R142 from saved records without training or remote computation."""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
ORDER = ['r142_analyze_original_control.py', 'r142_original_figures.py',
         'r142_extended_figures.py', 'r142_analyze_compatibility.py',
         'r142_integrate_manuscript.py', 'r142_release_documents.py',
         'r142_typeset.py', 'r142_verify_release.py', 'r142_release_audit.py']
if __name__ == '__main__':
    for script in ORDER:
        print(f'Rebuilding {script}', flush=True)
        subprocess.run([sys.executable, str(ROOT/'reproducibility'/script)], cwd=ROOT, check=True)
