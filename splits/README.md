# GBE participant split

A participant-level split for development and later evaluation. All plays from a person stay in the same set.

| Set | Participants |
|---|---:|
| Development | 10,000 |
| Hold-out | 37,067 |

- **Method:** NumPy `default_rng(20261007)`, sampling 10,000 IDs without replacement from IDs 1–47,067. Generation used NumPy 2.2.6.
- **File:** `gbe_split_2026-10-07.csv`, columns `id` and `set`.
- **SHA-256:** `861de3c444f506c76cbac3208840fd62d57a71b6aad63fceb83a802d8da139e9`.
- **Use:** further model development and reward–mood checks use development participants. The hold-out’s final evaluation role is still to be confirmed. Earlier descriptive and measurement checks used the full release, so it is not completely unseen.
- **Exclusions:** separate from the split; no exclusion rules have been finalised.

The CSV here is the copy loaded by the notebooks. Raw trial data stay outside the repository.

`make_gbe_split.py` records how the assignments were generated. It writes to the current directory; do not rerun it over the committed CSV. Keep these assignments fixed unless a change is agreed and recorded.
