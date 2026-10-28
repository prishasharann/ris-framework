# Phase 1 (ideal RIS) — 3-line summary for D

Setup: 28 GHz, 100 MHz, 40 dBm, noise -85 dBm, N = 1,024 / 4,096 / 9,216 / 16,384, 441-point UE grid, 2 buildings, 6 candidate RIS sites x 16 orientations (384 setups in `results_A.csv`).

1. Without a RIS, 56.5% of the grid has SNR >= 10 dB, and the shadow behind the buildings sits around -5 dB mean SNR; with the best placement (S3 at (65, 25, 12) m, normal pointing -y, 20 deg downtilt) coverage rises to 64.0% / 82.8% / 82.8% / 82.8% for N = 1,024 / 4,096 / 9,216 / 16,384.
2. Mean SNR gain grows steadily with N (+3.7, +6.7, +8.7, +10.3 dB), and shadow-zone coverage goes from 0% (no RIS) to 17% / 60% / 60% / 60%, so coverage saturates at N = 4,096 while SNR keeps rising; the remaining ~17% of the grid is behind B2 or out of the RIS's reach.
3. Placement matters more than N: the RIS must face the shadow it serves (S3 and S4 on the sides of B1 are best; S5, far away and behind B1, is worst at 56%), and with ideal phases SNR with the RIS is never below SNR without it (min gain = 0 dB over all 384 setups).

Figures: `maps/snr_maps_best_placement.png`, `maps/coverage_maps_best_placement.png`, `maps/snr_gain_map_N16384.png`, `maps/placement_heatmap.png`, `maps/coverage_snr_vs_N_per_site.png`
Table source: `summary_best_placement_A.csv` (best placement, per N), `placement_ranking_A.csv`, `results_A.csv` (all setups).

Notes for the paper: the model is free-space + a flat 40 dB blockage penalty per building crossing, spherical-wave (near-field) per-element distances, cos-pattern elements. It is not a ray-traced scene; B's results are the realistic check. No random trials were run in Phase 1 (deterministic), so there are no p-values from A.
