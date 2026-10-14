# K-means Dataset Lab report

The earlier customer demo has been replaced with dataset-driven clustering. Three independent CSV files supply unlabeled numeric observations. The user chooses K; no centers are supplied to training or stored as pretrained output.

The independent dist/kmeans.js module fits K-means using dataset-based K-means++ initialization, nearest-center assignments and mean updates. Optional standardization learns its statistics from the input rows. Several reproducible starting attempts are evaluated; the smallest-inertia run is retained. Empty clusters are repaired, duplicate feature rows are supported, and invalid K values are rejected.

Training runs in a browser Web Worker. The interactive interface supports built-in CSV selection, custom upload, arbitrary valid K, parameter changes, learning replay and point inspection. It displays learned centers, counts, inertia and all assignments. For each selected or new point, it compares every center distance, identifies the smallest, and shows both the distance equation and the mean calculation behind that center. SVG, center CSV and assignment CSV exports are available.

K-means is unsupervised: cluster numbers are arbitrary identifiers, not predicted semantic class labels. Inertia measures within-cluster squared distance and is not accuracy. Multiple initializations do not guarantee a global optimum. Iteration-capped results are marked as approximate.

See README.md for startup, CSV schemas, parameters and verification. The automated suite checks learning against known means, sensitivity to changed data, several K values on all datasets, scaling and solver behavior, plus the CSV-to-worker-to-interface flow.
