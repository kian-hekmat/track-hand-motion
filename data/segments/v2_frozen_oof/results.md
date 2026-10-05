# Phase 2 results: v2_frozen_oof (boundary tolerance 0.10 s)

Reported per group; never pooled. Frame accuracy excludes frames inside declared out-of-frame intervals (separate rows). `balanced` = mean of per-label accuracies; `tolerant` = frames within the tolerance of a true boundary also count if they match either neighbouring label. `chance` = recall of boundary sets with the same count as predicted (uniform / mean of 1000 random).


## clean

| take | scope | true | pred | matched | recall | precision | MAE matched (s) | chance recall | frame acc | balanced | tolerant | majority baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vid1 | all | 18 | 18 | 11 | 0.611 | 0.611 | 0.029 | 0.11/0.10 | 0.949 | 0.933 | 0.986 | 0.266 |
| vid2 | all | 18 | 18 | 16 | 0.889 | 0.889 | 0.054 | 0.17/0.10 | 0.966 | 0.950 | 0.998 | 0.305 |
| vid3 | all | 18 | 18 | 16 | 0.889 | 0.889 | 0.033 | 0.00/0.13 | 0.966 | 0.966 | 0.999 | 0.253 |

Per-label frame accuracy (scope=all):

| take | REST | REACH | GRASP | HOLD | RELEASE | RETRACT |
|---|---|---|---|---|---|---|
| vid1 | 1.00 (n=256) | 0.96 (n=136) | 0.85 (n=160) | 0.99 (n=260) | 0.95 (n=63) | 0.84 (n=101) |
| vid2 | 0.98 (n=310) | 0.92 (n=135) | 0.99 (n=154) | 0.99 (n=255) | 0.86 (n=77) | 0.96 (n=84) |
| vid3 | 0.98 (n=178) | 1.00 (n=98) | 0.95 (n=162) | 0.96 (n=196) | 0.98 (n=59) | 0.91 (n=82) |

## fast

| take | scope | true | pred | matched | recall | precision | MAE matched (s) | chance recall | frame acc | balanced | tolerant | majority baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vid4 | all | 18 | 15 | 14 | 0.778 | 0.933 | 0.031 | 0.33/0.27 | 0.911 | 0.785 | 1.000 | 0.363 |

Per-label frame accuracy (scope=all):

| take | REST | REACH | GRASP | HOLD | RELEASE | RETRACT |
|---|---|---|---|---|---|---|
| vid4 | 0.99 (n=102) | 0.92 (n=38) | 0.83 (n=42) | 1.00 (n=57) | 0.00 (n=13) | 0.97 (n=29) |

## hard

| take | scope | true | pred | matched | recall | precision | MAE matched (s) | chance recall | frame acc | balanced | tolerant | majority baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vid5 | all | 24 | 27 | 18 | 0.750 | 0.667 | 0.035 | 0.21/0.16 | 0.841 | 0.831 | 0.876 | 0.258 |
| vid5 | out_of_frame | | | | | | | | 0.500 (20 frames) | | | |
| vid5 | cycle1 | 6 | 7 | 5 | 0.833 | 0.714 | 0.020 | 0.00/0.04 | 0.966 | 0.935 | 1.000 | 0.335 |
| vid5 | cycle2 | 6 | 5 | 3 | 0.500 | 0.600 | 0.044 | 0.00/0.03 | 0.866 | 0.875 | 0.902 | 0.361 |
| vid5 | cycle3 | 6 | 7 | 4 | 0.667 | 0.571 | 0.041 | 0.00/0.04 | 0.778 | 0.805 | 0.807 | 0.307 |
| vid5 | cycle4 | 6 | 8 | 4 | 0.667 | 0.500 | 0.048 | 0.33/0.05 | 0.782 | 0.845 | 0.817 | 0.454 |

Per-label frame accuracy (scope=all):

| take | REST | REACH | GRASP | HOLD | RELEASE | RETRACT |
|---|---|---|---|---|---|---|
| vid5 | 0.99 (n=230) | 0.59 (n=206) | 0.80 (n=173) | 0.97 (n=212) | 0.82 (n=17) | 0.82 (n=55) |
