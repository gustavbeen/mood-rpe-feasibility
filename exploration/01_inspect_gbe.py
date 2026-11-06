# GBE - looking at a couple of participants / plays for now
from pathlib import Path
import sys

import numpy as np

# use the same loader as the notebooks
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
from gbe_loader import load_gbe

participants = load_gbe()

print("Container type:", type(participants))
print("Number of records:", len(participants))

# first person, just to see what is in here
first_participant = participants[0]

print("Record type:", type(first_participant))
print("Fields:", first_participant.keys())

for field in ["id", "nPlays", "timesPlayed", "dayNumber", "designVersion"]:
    print(field, "=", first_participant[field])

task_data = first_participant["data"]

print("Task data type:", type(task_data))
print("Task data shape:", task_data.shape)

headers = first_participant["dataHdr"]

for index, name in enumerate(headers):
    print(f"Python index {index:2d}: {name}")

columns = [0, 2, 3, 4, 6, 7, 9]

print("\nSelected columns:")
print([headers[index] for index in columns])

print("\nFirst six rows:")
print(task_data[:6, columns])

# gamble EV = average of the two outcomes (50/50)
# certain rewards go in their own term
offered_ev = (task_data[:, 3] + task_data[:, 4]) / 2
print("\nOffered EV:")
print(offered_ev[:3])

chose_gamble = task_data[:, 6] == 1
print("\nChose gamble:")
print(chose_gamble[:3])

chosen_ev = np.where(chose_gamble, offered_ev, 0)
print("\nChosen EV:")
print(chosen_ev[:3])

certain_reward = np.where(chose_gamble, 0, task_data[:, 2])
print("\nCertain reward:")
print(certain_reward[:3])

rpe = np.where(chose_gamble, task_data[:, 7] - offered_ev, 0)
print("\nRPE:")
print(rpe[:3])

# quick check on paper:
# trial 1: (0 - 220)/2 = -110; outcome -220
#          RPE = -220 - (-110) = -110
# trial 2: chose certain 45 -> CR = 45, gamble EV and RPE terms = 0
#          offered gamble EV would be 122/2 = 61
# trial 3: 132/2 = 66; got 132 -> RPE = 132 - 66 = 66
np.testing.assert_allclose(chosen_ev[:3], [-110, 0, 66])
np.testing.assert_allclose(certain_reward[:3], [0, 45, 0])
np.testing.assert_allclose(rpe[:3], [-110, 0, 66])

print("First three trials match the hand calculations.")

# outcome 0 does not mean RPE 0!
zero_outcome_gambles = chose_gamble & (task_data[:, 7] == 0)

print("\nZero-outcome gamble trial numbers:")
print(task_data[zero_outcome_gambles, 0])

print("Offers and outcomes:")
print(task_data[zero_outcome_gambles][:, [0, 3, 4, 7]])

print("Calculated RPEs:")
print(rpe[zero_outcome_gambles])

# trial 8 (index 7): EV = -113/2 = -56.5; RPE = 0 - (-56.5) = +56.5
# trial 12 (index 11): EV = 132/2 = 66; RPE = 0 - 66 = -66
np.testing.assert_allclose(offered_ev[[7, 11]], [-56.5, 66])
np.testing.assert_allclose(rpe[[7, 11]], [56.5, -66])

# where are the happiness ratings?
# dictionary: rating in row 1 is before the task
happiness = task_data[:, 9]
has_rating = ~np.isnan(happiness)

print("Rows containing ratings, expressed as task trial numbers:")
print(task_data[has_rating, 0])

print("Happiness values:")
print(happiness[has_rating])

rating_boundaries = task_data[has_rating, 0].astype(int).copy()
rating_boundaries[0] = 0  # baseline = 0 completed trials, although stored in row 1

print("Rating boundaries:", rating_boundaries)
print("Trials between ratings:", np.diff(rating_boundaries))

# ratings at 9 and 11: happiness 71 -> 50
# so look at trials 10, 11. Both certain choices, +45 then -55
# leave the events separate for now - aggregation still to decide
left = 9
right = 11

print("\nInterval after trial 9 through trial 11:")

print("Trial numbers:", task_data[left:right, 0])
print("Choices:", task_data[left:right, 6])
print("Outcomes:", task_data[left:right, 7])
print("RPEs:", rpe[left:right])

np.testing.assert_array_equal(task_data[left:right, 0], [10, 11])
np.testing.assert_allclose(certain_reward[left:right], [45, -55])
np.testing.assert_allclose(chosen_ev[left:right], [0, 0])
np.testing.assert_allclose(rpe[left:right], [0, 0])

# someone with repeat plays next
# need to go down another level to get the trial matrix
for participant in participants:
    if participant["nPlays"] > 1:
        repeat_participant = participant
        break

print("\nRepeat participant ID:", repeat_participant["id"])
print("Available plays:", repeat_participant["nPlays"])
print("Data type:", type(repeat_participant["data"]))
print("Data shape:", repeat_participant["data"].shape)

repeat_plays = repeat_participant["data"]
first_repeat_play = repeat_plays[0]

print("First play type:", type(first_repeat_play))
print("First play shape:", first_repeat_play.shape)

for field in ["timesPlayed", "dayNumber", "timeOfDay", "designVersion"]:
    print(field, "=", repeat_participant[field])

assert len(repeat_plays) == repeat_participant["nPlays"]

for field in ["timesPlayed", "dayNumber", "timeOfDay", "designVersion"]:
    assert len(repeat_participant[field]) == len(repeat_plays)

# match each play to its metadata; trial numbering starts again at 1
for play_index, play in enumerate(repeat_plays):
    print(
        "Available play:", play_index + 1,
        "| Started attempt:", repeat_participant["timesPlayed"][play_index],
        "| Shape:", play.shape,
        "| First/last trial:", play[0, 0], play[-1, 0],
    )