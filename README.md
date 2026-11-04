# Mood × RPE feasibility work

Exploratory feasibility work for an MSc Psychology dissertation (Arden University). The research question is whether prior mood moderates the association between subsequent reward prediction errors (RPEs) and later momentary happiness, within individuals.

These notebooks reconstruct and check the Great Brain Experiment (GBE) risky-choice data, look at a slider measurement issue, implement the happiness model from Rutledge et al. (2014), and test a provisional mood × RPE model on synthetic data. No mood × RPE estimate has been made on real data, and the hold-out participants are reserved for the main analysis.

## Data

The GBE risk task data are available from Dryad (doi:10.5061/dryad.prr4xgxkk) under CC0. The raw data are not included in this repository. The notebooks expect the file at:

```
../data/raw/gbe/Rutledge_GBE_risk_data_TOD.mat
```

relative to the repository root, i.e. in a `data/` folder next to the repository.

## Setup

Python 3.12 or later. Install the packages in `requirements.txt` and run the notebooks from the `notebooks/` folder.

## Contents

| Notebook                    | Content                                                                           |
| --------------------------- | --------------------------------------------------------------------------------- |
| `01_understand_gbe`         | Data structure and reward-term reconstruction for example plays                   |
| `02_validate_gbe`           | Dataset-wide structural checks, constant ratings and slider starting values       |
| `03_gbe_slider_measurement` | Slider starts, response times and ratings that match the start value              |
| `04_gbe_reward_intervals`   | Reward intervals between ratings; matching and reward exposure                    |
| `05_gbe_happiness_model`    | Implementation of the Rutledge et al. (2014) happiness model across task versions |
| `06_mood_rpe_simulation`    | Provisional mood × RPE model and recovery checks on synthetic data                |

`splits/` holds the fixed participant-level development/hold-out split and the script that generated it (see `splits/README.md`). `exploration/` holds an early inspection script.

## Reference

Rutledge, R. B., Skandali, N., Dayan, P., & Dolan, R. J. (2014). A computational and neural model of momentary subjective well-being. _Proceedings of the National Academy of Sciences, 111_(33), 12252–12257.
