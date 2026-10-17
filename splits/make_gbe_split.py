"""Create the GBE development / hold-out split (run once, 7 Oct 2026).

Split is by participant, so all plays of a participant fall in the same set.
GBE participant ids are sequential integers 1..47,067 (verified in
'Data Validation and Repository Notes.md'), so the split needs only the ids,
not the data. No outcome or rating data were read to make it.
"""
import csv, hashlib
import numpy as np

SEED = 20261007
N_PARTICIPANTS = 47067
N_DEVELOPMENT = 10000

rng = np.random.default_rng(SEED)
ids = np.arange(1, N_PARTICIPANTS + 1)
dev = set(rng.choice(ids, size=N_DEVELOPMENT, replace=False).tolist())

out = "gbe_split_2026-10-07.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "set"])
    for i in ids:
        w.writerow([int(i), "development" if int(i) in dev else "holdout"])

print("development:", len(dev), "holdout:", N_PARTICIPANTS - len(dev))
print("sha256:", hashlib.sha256(open(out, "rb").read()).hexdigest())
