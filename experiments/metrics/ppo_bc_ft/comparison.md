| Hunter | Win rate | Escape rate | Timeout | Time to capture | Interceptions | False investigations/ep |
|---|---|---|---|---|---|---|
| random | 26.1% ± 3.9% | 56.1% | 17.8% | 142.9 steps | 6.7% | 0.21 |
| patrol_chaser | 88.3% ± 1.4% | 10.0% | 1.7% | 62.1 steps | 5.0% | 0.50 |
| habit_heuristic | 90.0% ± 4.9% | 7.8% | 2.2% | 40.4 steps | 6.1% | 0.19 |
| ppo_scratch_v1 | 75.6% ± 3.9% | 23.3% | 1.1% | 35.5 steps | 7.2% | 0.11 |
| imitation_only (bc_init) | 61.7% ± 2.4% | 23.3% | 15.0% | 47.9 steps | 17.2% | 0.15 |
| ppo_early (ckpt_25000_steps) | 66.7% ± 2.7% | 19.4% | 13.9% | 47.0 steps | 19.4% | 0.12 |
| ppo_final | 71.7% ± 4.9% | 17.8% | 10.6% | 51.1 steps | 22.2% | 0.16 |
| ppo_best | 63.9% ± 3.9% | 24.4% | 11.7% | 51.9 steps | 19.4% | 0.17 |

| Hunter | creature_of_habit | ghost | hider | runner | trickster | unpredictable | wanderer |
|---|---|---|---|---|---|---|---|
| random | 13% | 17% | 0% | 49% | 35% | 27% | 35% |
| patrol_chaser | 84% | 76% | 79% | 100% | 93% | 92% | 88% |
| habit_heuristic | 92% | 83% | 87% | 94% | 84% | 91% | 100% |
| ppo_scratch_v1 | 87% | 84% | 92% | 61% | 77% | 65% | 65% |
| imitation_only (bc_init) | 49% | 55% | 52% | 85% | 54% | 76% | 70% |
| ppo_early (ckpt_25000_steps) | 66% | 55% | 46% | 78% | 62% | 76% | 83% |
| ppo_final | 81% | 46% | 49% | 64% | 79% | 89% | 80% |
| ppo_best | 66% | 52% | 52% | 78% | 66% | 74% | 64% |
