# LPF Known Issues

---

## Open Issues

### Infrastructure

- [ ] **CI/CD not configured** — No `.github/workflows/`, `.travis.yml`, or `tox.ini` for automated testing.
- [ ] **Shell scripts broken** — `run_evosearch_all.sh` etc. reference non-existent Python files.
- [ ] **Shell scripts missing shebang** (`#!/bin/bash`)
- [ ] **`.gitignore` contains irrelevant entries** — Django, Flask patterns included.
- [ ] **`search/run_evosearch_succinea.psh`** — Wrong file extension `.psh`.

### Design / Architecture

- [ ] **`not x` → `x is None` replacement needed** — Most widespread bug pattern (40+ locations). Valid falsy values (0, 0.0, "") silently replaced with defaults.
- [ ] **Heavy code duplication across models** — `to_dict()`, `parse_params()`, `get_param_bounds()` are nearly identical. Could be driven by a class-level parameter descriptor.
- [ ] **`SolverFactory` substring matching fragility** — `if "x" in name:` should be replaced with exact-match dict lookup.
- [ ] **`ConverterFactory` only supports Liaw** — Other converters and `TwoComponentCrosstalkDiploidModel` are unregistered.
- [ ] **VGG16 model loaded 4 times** (`lpf/objectives/vggperceptualloss.py:29-32`)
- [ ] **Unbounded cache growth in evolutionary search** (`lpf/search/evosearch.py:91`) — Should use LRU cache.
- [ ] **Hard-coded image sizes in `image.py`** — Resize/crop coordinates are magic numbers.
- [ ] **Inconsistent optional dependency import handling** — `vggperceptualloss.py`, `perceptualsimilarity.py`.
- [ ] **Dead code in `rdmodel.py` `y_mesh` setter** — `if self._y_mesh is None:` branch unreachable.
- [ ] **`get_param_bounds()` range calculation error** — `range(N, 2 * n_init_pts, 2)` should be `range(N, N + 2 * n_init_pts, 2)`.

### Minor

- [ ] **Dead code in `solverfactory.py:1-19`** — Commented-out imports.
- [ ] **`solverfactory.py:72`** — Recommends explicit method for stiff problems (implicit method needed).
- [ ] **`adamsbashforth2solver.py`** — `_prev_t` stored but never read.
