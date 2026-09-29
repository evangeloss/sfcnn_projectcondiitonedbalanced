# Preserved additional tests

Everything added after the original SFCNN section is kept in this folder, in source-notebook order. The contents include:

- NMSE versus number of propagation paths;
- plots comparing recorded NMSE results;
- an alternate array-size training run;
- an MMSE baseline experiment;
- direct received-pilot `Y -> H0` dataset/training experiments;
- achievable-rate evaluation;
- FIM-based deformation-codebook scoring and greedy optimization;
- single-view and eight-view training runs;
- morphing-range sweeps and deformation-range checks.

Run `run_additional_tests.py` from the repository root to bootstrap the original cells and then execute these tests. The full sequence is expensive and contains multiple training runs. Each numbered file may also be opened independently for inspection or selective reuse, but most expect state created by earlier notebook cells.
