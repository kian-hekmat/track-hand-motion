# Phase 2 results: v1_rules_pelt (boundary tolerance 0.10 s)

Reported per group; never pooled. Frame accuracy excludes frames inside declared out-of-frame intervals (separate rows). `balanced` = mean of per-label accuracies; `tolerant` = frames within the tolerance of a true boundary also count if they match either neighbouring label. `chance` = recall of boundary sets with the same count as predicted (uniform / mean of 1000 random).


## clean

| take | scope | true | pred | matched | recall | precision | MAE matched (s) | chance recall | frame acc | balanced | tolerant | majority baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vid1 | all | 18 | 13 | 9 | 0.500 | 0.692 | 0.033 | 0.06/0.07 | 0.680 | 0.708 | 0.717 | 0.266 |
| vid2 | all | 18 | 18 | 14 | 0.778 | 0.778 | 0.041 | 0.17/0.10 | 0.778 | 0.737 | 0.803 | 0.305 |
| vid3 | all | 18 | 17 | 11 | 0.611 | 0.647 | 0.028 | 0.17/0.12 | 0.592 | 0.615 | 0.628 | 0.253 |

Per-label frame accuracy (scope=all):

| take | REST | REACH | GRASP | HOLD | RELEASE | RETRACT |
|---|---|---|---|---|---|---|
| vid1 | 1.00 (n=256) | 0.97 (n=136) | 0.26 (n=160) | 0.37 (n=260) | 0.75 (n=63) | 0.90 (n=101) |
| vid2 | 0.99 (n=310) | 0.99 (n=135) | 0.34 (n=154) | 0.72 (n=255) | 0.42 (n=77) | 0.96 (n=84) |
| vid3 | 0.99 (n=178) | 0.65 (n=98) | 0.55 (n=162) | 0.15 (n=196) | 0.44 (n=59) | 0.90 (n=82) |

## fast

| take | scope | true | pred | matched | recall | precision | MAE matched (s) | chance recall | frame acc | balanced | tolerant | majority baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vid4 | all | 18 | 6 | 6 | 0.333 | 1.000 | 0.022 | 0.17/0.12 | 0.452 | 0.312 | 0.498 | 0.363 |

Per-label frame accuracy (scope=all):

| take | REST | REACH | GRASP | HOLD | RELEASE | RETRACT |
|---|---|---|---|---|---|---|
| vid4 | 0.96 (n=102) | 0.29 (n=38) | 0.00 (n=42) | 0.00 (n=57) | 0.00 (n=13) | 0.62 (n=29) |

## hard

| take | scope | true | pred | matched | recall | precision | MAE matched (s) | chance recall | frame acc | balanced | tolerant | majority baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vid5 | all | 24 | 22 | 13 | 0.542 | 0.591 | 0.029 | 0.04/0.13 | 0.542 | 0.490 | 0.579 | 0.258 |
| vid5 | out_of_frame | | | | | | | | 0.250 (20 frames) | | | |
| vid5 | cycle1 | 6 | 6 | 3 | 0.500 | 0.500 | 0.011 | 0.00/0.04 | 0.729 | 0.585 | 0.764 | 0.335 |
| vid5 | cycle2 | 6 | 5 | 2 | 0.333 | 0.400 | 0.018 | 0.00/0.03 | 0.861 | 0.601 | 0.902 | 0.361 |
| vid5 | cycle3 | 6 | 6 | 2 | 0.333 | 0.333 | 0.067 | 0.00/0.04 | 0.456 | 0.383 | 0.485 | 0.307 |
| vid5 | cycle4 | 6 | 5 | 3 | 0.500 | 0.600 | 0.035 | 0.00/0.03 | 0.214 | 0.447 | 0.249 | 0.454 |

Per-label frame accuracy (scope=all):

| take | REST | REACH | GRASP | HOLD | RELEASE | RETRACT |
|---|---|---|---|---|---|---|
| vid5 | 0.98 (n=230) | 0.72 (n=206) | 0.02 (n=173) | 0.25 (n=212) | 0.00 (n=17) | 0.96 (n=55) |
