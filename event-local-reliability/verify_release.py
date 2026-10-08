"""Verify the frozen numeric release without fitting models or downloading data."""
from pathlib import Path
import hashlib
import numpy as np
root = Path(__file__).resolve().parent
for line in (root / "SHA256SUMS.txt").read_text().splitlines():
    expected, name = line.split("  ", 1)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
for name, expected in [("source_data", (16384, 30)), ("public_validation", (18432, 19))]:
    with np.load(root / name / "reliability_features.npz", allow_pickle=False) as z:
        assert set(z.files) == {"X", "names", "global_n"}
        assert z["X"].shape == expected and np.isfinite(z["X"]).all()
        assert len(z["names"]) == expected[1]
print("PASS: included file hashes and numeric feature schemas")
