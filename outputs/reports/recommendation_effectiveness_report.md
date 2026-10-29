# Recommendation effectiveness report

> EPANET simulated validation. This validates the recommendation in the simulated EPANET network. It is not field validation in an actual water utility.
> All data in this project is synthetic: it comes only from EPANET simulations run through WNTR. No real utility data is used.

23 recommendations (one decision window each, chosen as the highest-severity window of different test scenarios per situation) were each re-simulated in EPANET before and after every bounded candidate action (valve −15…+15 %, pump −10…+10 %). The lowest-change safe action with the best objective is kept; if none is safe the result is "No safe action found; operator inspection recommended."

## Summary by situation

| group | n_recommendations | n_validated_improvement | n_improvement_resolved | n_improvement_partial | n_no_action_needed | n_no_safe_action_found | n_inspection_recommended | mean_low_violations_before | mean_low_violations_after | mean_high_violations_before | mean_high_violations_after | mean_spread_before_m | mean_spread_after_m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| burst_like_pattern | 2 | 0 | 0 | 0 | 0 | 0 | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 9.45 | 9.45 |
| critical_low | 4 | 2 | 1 | 1 | 1 | 1 | 0 | 3.00 | 1.25 | 0.00 | 0.00 | 11.99 | 11.99 |
| high | 4 | 4 | 0 | 4 | 0 | 0 | 0 | 0.00 | 0.00 | 8.00 | 8.00 | 35.55 | 33.25 |
| imbalance | 4 | 4 | 0 | 4 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 31.27 | 27.92 |
| leak_like_pattern | 3 | 0 | 0 | 0 | 0 | 0 | 3 | 0.00 | 0.00 | 0.00 | 0.00 | 8.52 | 8.52 |
| normal | 4 | 0 | 0 | 0 | 4 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 7.18 | 7.18 |
| other_low | 2 | 1 | 0 | 1 | 0 | 1 | 0 | 4.00 | 2.00 | 0.00 | 0.00 | 35.14 | 33.74 |
| ALL | 23 | 11 | 1 | 10 | 5 | 2 | 5 | 0.87 | 0.39 | 1.39 | 1.39 | 19.95 | 18.84 |

## Case by case

| Scenario | Decision time | Engine problem | Chosen action | Result | Level | Low viol. before | after | High viol. before | after  | Spread before (m) | after   | J4 min before (m) | after    | Loss proxy (L/s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0015 | 2026-01-05 02:45:00 | normal | no change | no_action_needed | not_applicable | 0 | 0 | 0 | 0 | 6.09 | 6.09 | 46.89 | 46.89 | 0.00 |
| S0051 | 2026-01-05 16:15:00 | normal | no change | no_action_needed | not_applicable | 0 | 0 | 0 | 0 | 11.04 | 11.04 | 50.20 | 50.20 | 0.00 |
| S0056 | 2026-01-05 02:45:00 | normal | no change | no_action_needed | not_applicable | 0 | 0 | 0 | 0 | 5.02 | 5.02 | 48.07 | 48.07 | 0.00 |
| S0060 | 2026-01-05 14:15:00 | normal | no change | no_action_needed | not_applicable | 0 | 0 | 0 | 0 | 6.59 | 6.59 | 37.69 | 37.69 | 0.00 |
| S0113 | 2026-01-05 02:45:00 | high | pump -10% | validated_improvement | partial | 0 | 0 | 8 | 8 | 35.41 | 33.25 | 77.72 | 75.56 | 0.00 |
| S0147 | 2026-01-05 02:45:00 | high | pump -10% | validated_improvement | partial | 0 | 0 | 8 | 8 | 44.78 | 42.42 | 85.95 | 83.59 | 0.00 |
| S0195 | 2026-01-05 04:00:00 | high | pump -10% | validated_improvement | partial | 0 | 0 | 8 | 8 | 16.51 | 14.28 | 64.73 | 62.51 | 0.00 |
| S0209 | 2026-01-05 02:45:00 | high | pump -10% | validated_improvement | partial | 0 | 0 | 8 | 8 | 45.51 | 43.07 | 83.67 | 81.23 | 0.00 |
| S0060 | 2026-01-05 14:00:00 | critical_low | no change | no_action_needed | not_applicable | 0 | 0 | 0 | 0 | 8.59 | 8.59 | 30.44 | 30.44 | 0.00 |
| S0105 | 2026-01-05 09:15:00 | critical_low | pump +10% | validated_improvement | resolved | 4 | 0 | 0 | 0 | 13.19 | 13.19 | 29.14 | 30.88 | 0.00 |
| S0124 | 2026-01-05 09:15:00 | critical_low | pump +10% | validated_improvement | partial | 4 | 1 | 0 | 0 | 13.18 | 13.18 | 27.75 | 29.71 | 0.00 |
| S0140 | 2026-01-05 17:15:00 | critical_low | no change | no_safe_action_found | none | 4 | 4 | 0 | 0 | 13.01 | 13.01 | 27.18 | 27.18 | 0.00 |
| S0015 | 2026-01-05 12:45:00 | other_low | no change | no_safe_action_found | none | 4 | 4 | 0 | 0 | 33.82 | 33.82 | 46.89 | 46.89 | 0.00 |
| S0194 | 2026-01-05 15:30:00 | other_low | valve +15% | validated_improvement | partial | 4 | 0 | 0 | 0 | 36.46 | 33.66 | 49.89 | 49.86 | 0.00 |
| S0077 | 2026-01-05 03:45:00 | imbalance | valve +15% | validated_improvement | partial | 0 | 0 | 0 | 0 | 27.10 | 23.64 | 45.09 | 45.09 | 0.00 |
| S0128 | 2026-01-05 15:30:00 | imbalance | valve +15% | validated_improvement | partial | 0 | 0 | 0 | 0 | 32.46 | 28.92 | 50.90 | 50.90 | 0.00 |
| S0198 | 2026-01-05 23:15:00 | imbalance | valve +15% | validated_improvement | partial | 0 | 0 | 0 | 0 | 34.84 | 31.70 | 50.67 | 50.67 | 0.00 |
| S0218 | 2026-01-05 15:45:00 | imbalance | valve +15% | validated_improvement | partial | 0 | 0 | 0 | 0 | 30.68 | 27.41 | 47.46 | 47.46 | 0.00 |
| S0083 | 2026-01-05 16:45:00 | leak_like_pattern | no change | inspection_recommended | not_applicable | 0 | 0 | 0 | 0 | 11.45 | 11.45 | 41.82 | 41.82 | 2.97 |
| S0176 | 2026-01-05 16:45:00 | leak_like_pattern | no change | inspection_recommended | not_applicable | 0 | 0 | 0 | 0 | 8.50 | 8.50 | 38.87 | 38.87 | 2.09 |
| S0229 | 2026-01-05 15:45:00 | leak_like_pattern | no change | inspection_recommended | not_applicable | 0 | 0 | 0 | 0 | 5.61 | 5.61 | 45.97 | 45.97 | 2.77 |
| S0158 | 2026-01-05 02:45:00 | burst_like_pattern | no change | inspection_recommended | not_applicable | 0 | 0 | 0 | 0 | 5.01 | 5.01 | 45.13 | 45.13 | 7.32 |
| S0268 | 2026-01-05 11:00:00 | burst_like_pattern | no change | inspection_recommended | not_applicable | 0 | 0 | 0 | 0 | 13.89 | 13.89 | 39.55 | 39.55 | 7.05 |

## Honest reading

- 11 of 23 cases ended in a validated improvement; 10 of those only partly cleared the problem (e.g. high source head: pressure falls but violations remain).
- 2 case(s) ended with no safe action; the engine's first-order estimate can be wrong where the real cause is a blockage.
- Leak-like and burst-like patterns are inspection-only: no pump/valve change is tried because that could hide a leak. The loss proxy (inflow minus delivered demand) is the leak outflow in the simulation.
- A forecast false alarm shows up as "no violation in EPANET": the model predicted a problem that did not occur.
- Improvement is measured over the next hour in the same simulator that generated the data; it says nothing about a real network.
