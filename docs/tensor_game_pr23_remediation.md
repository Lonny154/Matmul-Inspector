# PR #23 remediation baseline

Baseline captured on October 15, 2026 at commit `f52e200` on
`research/tensor-game`.

- Local HEAD matched the remote PR head. The PR was mergeable and had no newer
  commits, comments, submitted reviews, or conflicts.
- Existing GitHub checks passed, but CTest stopped before the bounded-memory,
  provider-integration, release-gate, and cross-machine tests.
- The artifact-populated checkout passed the five focused files: 17 tests.
- A clean `git archive` reproduced four failures. One bounded-memory test needed
  ignored TG-4I/TG-4S data. Three TG-4X tests needed the ignored frozen bundle.
- The public factory default was `bounded_128`. Explicit `exhaustive` selection
  worked for `(4, 4, 4)` tensors.
- A top-level JSON list was rejected. A `(1, 1, 1)` target reached fixed-size
  provider indexing instead of receiving a supported-shape error.
- `max_wall_time_seconds` was checked only before simulations. It was not a hard
  execution deadline.

The remediation keeps candidate scoring and ordering unchanged. Production
provider construction now has a stable `tensor_game_residual` facade. The older
`tensor_game_bounded_residual` and `tensor_game_search_calibration` imports remain
available as compatibility paths for research scripts.
