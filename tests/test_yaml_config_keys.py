"""YAML config files must have correct key names and required keys.

Regression tests for Section 3 issues:
- 3-1: COLOV_V typo → COLOR_V
- 3-2: NUM_INIT_PTS → N_INIT_PTS
- 3-3: spectabilis config referencing wrong init_pop
- 3-4: CPU configs missing MODEL and SOLVER keys
"""

import os
import pytest
import yaml


CONFIG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "config", "evosearch"
)

# Keys required by all species-specific configs
REQUIRED_KEYS = {
    "N_PROCS", "N_GEN", "DX", "DT", "WIDTH", "HEIGHT",
    "THR_COLOR", "N_INIT_PTS", "N_ITERS", "RTOL_EARLY_STOP",
    "POP_SIZE", "EVAL_INIT_FITNESS", "INIT_POP", "INITIALIZER",
    "MODEL", "SOLVER", "OBJECTIVES",
    "LADYBIRD_TYPE", "LADYBIRD_SUBTYPES", "DPATH_OUTPUT",
}


def _load_yaml(filename):
    fpath = os.path.join(CONFIG_DIR, filename)
    if not os.path.exists(fpath):
        pytest.skip(f"{filename} not found")
    with open(fpath, "r") as f:
        return yaml.safe_load(f)


def _all_yaml_files():
    if not os.path.isdir(CONFIG_DIR):
        return []
    return [f for f in os.listdir(CONFIG_DIR) if f.endswith(".yaml")]


class TestYamlConfigRequiredKeys:
    """Species-specific configs must have all required keys."""

    @pytest.mark.parametrize("filename", [
        f for f in _all_yaml_files() if "all" not in f
    ])
    def test_species_config_has_required_keys(self, filename):
        cfg = _load_yaml(filename)
        missing = REQUIRED_KEYS - set(cfg.keys())
        assert not missing, (
            f"{filename} is missing required key(s): {missing}"
        )

    def test_ahexaspilota_has_color_v(self):
        """ahexaspilota defines custom colors — COLOR_V must be present."""
        cfg = _load_yaml("config_search_ahexaspilota_gpu.yaml")
        assert "COLOR_V" in cfg


class TestYamlConfigInitPop:
    """INIT_POP should reference the correct species population."""

    @pytest.mark.parametrize("species", [
        "spectabilis", "succinea", "axyridis", "ahexaspilota",
    ])
    def test_init_pop_matches_species(self, species):
        cfg = _load_yaml(f"config_search_{species}_gpu.yaml")
        assert species in cfg["INIT_POP"], (
            f"{species} config should reference init_pop_{species}, "
            f"got {cfg['INIT_POP']}"
        )
