# Tests

Tests are maintained regression coverage for current terrain models, data processing,
training, inference, exports, and demo behavior. They live outside `src/` and use
temporary data or small synthetic models unless a group explicitly requires real
ONNX exports. Shared Python/web contract fixtures live in `fixtures/`.

```text
tests/
  config/                        path overrides, services, and CLI configuration
  demos/                         image demo and palette behavior
  etopo/                         analysis, normalization, synthetic levels, reports
  inference/                     geometry, noise, cache, models, and world generation
  models/                        shared upscaling and noise clipping
  packaging/                     standalone inference release staging
  projects/z0_generator/         generator training and export
  projects/zn_map_upscaler/       input maps, training, logging, profiling, export
  training/                      shared losses
  fixtures/                      versioned cross-runtime contract data
  web/unit/                      model-free functions, controllers, and mesh invariants
  web/integration/               cross-component scene checks
  web/models/                    actual ONNX inference and model contracts
  web/browser/                   browser controls, rendering, and navigation
  web/support/                   shared loaders and browser input assertions
```

## Python

Use the project's configured interpreter. Run CUDA validation on an explicitly
selected CUDA machine. Install the test extra into that development environment
with its package manager; for a pip environment:

```sh
python -m pip install -e '.[test]'
python -m pytest
```

Pytest discovers tests under `tests/`, adds `src/` to the import path, and uses
importlib mode so subsystems can use names such as `test_models.py` independently.
Existing `unittest.TestCase` classes are run by pytest alongside fixture-based tests;
tests must not import or manually execute other test modules.

Useful focused commands, from the repository root:

```sh
python -m pytest tests/inference tests/demos tests/packaging
python -m pytest tests/projects/zn_map_upscaler
python -m pytest -m cuda
python -m pytest -m export
```

CUDA-specific cases skip when CUDA is unavailable. Run the complete suite on the
CUDA host before handing off shared training/model changes. The `export` group
checks actual PyTorch/ONNX artifacts, dynamic shapes, and model state preservation.
No training datasets or durable model checkpoints are required by the Python suite.

## Web

Use Node 22+ and install the demo's pinned development dependencies:

```sh
cd src/terrain_higan/demos/web
npm ci
npm test
```

Tests resolve their runtime dependencies from the demo's `node_modules/`; there is
no second JavaScript package to install. `npm test` and `npm run test:unit` run the
quick, model-free unit group. Other groups are explicit:

```sh
npm run test:integration
npm run sync-models
npm run test:models
npm run test:browser
npm run test:all
```

Integration checks take longer than unit checks.
Model checks require the developer's current ONNX exports in `.local/weights/`
(or `--weights-dir /path/to/weights`) and their synchronized browser copies. The runner
checks model freshness before model/browser groups. Python-only and unit-only
changes do not require synchronizing ONNX models.

Browser checks use Playwright-managed Chromium: run `npx playwright install chromium`
in the demo directory after `npm ci`. Set `CHROME_PATH` to use an installed browser.
The runner starts and stops a loopback demo server on
port 4189 automatically. Set `TEST_PORT` to choose another port, or `DEMO_URL` to
use an existing server.

Select individual scripts by appending their names:

```sh
npm run test:browser -- test-browser.cjs test-globe-browser.cjs
npm run test:models -- test-worker.cjs
SEAM_ONLY=1 npm run test:browser -- test-browser.cjs
npm run format:check
```

The runner discovers entry points in each selected group, runs each in its own
process, continues after failures, and exits nonzero if any script fails. Screenshots
and exports go under ignored `runs/` directories; shared fixtures are committed.

## Maintaining coverage

- Add a regression to the relevant subsystem when fixing a behavioral bug.
- Assert shapes, coordinate/noise identity, conservation, gradients, loss behavior,
  export contracts, or visible interactions. Avoid pinning tunable training defaults
  or exact diagnostic timing thresholds.
- Consolidate checks for the same behavior instead of adding a new one-off file.
- Remove tests for deleted projects. Preserve shared assertions before removing
  a file containing both obsolete and current coverage.
- Keep debugging reproductions local until they become a maintained regression.
- Stage new tests and fixtures with the implementation; a passing local untracked
  test is not available to the next checkout or agent.
