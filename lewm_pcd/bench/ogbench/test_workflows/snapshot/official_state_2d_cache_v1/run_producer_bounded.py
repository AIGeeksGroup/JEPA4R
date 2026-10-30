"""Resource-only launcher; execute the frozen original producer CLI."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    import torch
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.35)
    source = Path(__file__).resolve().parent / 'produce.py'
    sys.argv[0] = str(source)
    runpy.run_path(str(source), run_name='__main__')
