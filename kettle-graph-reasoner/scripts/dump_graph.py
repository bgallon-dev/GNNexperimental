import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "src/data/corpus/tier1/graph_000000.npz"
d = np.load(path, allow_pickle=True)
np.set_printoptions(threshold=np.inf)
for k in d.files:
    a = d[k]
    print(f"--- {k} --- shape={getattr(a, 'shape', None)} dtype={getattr(a, 'dtype', None)}")
    print(a)
    print()
