"""Every esx module that was moved from another repository must equal its source file except for import paths.

esx was assembled by moving files verbatim (see DESIGN.md, "Provenance"). Numerics (noise from seed, sigma convention,
update rule) therefore equal the legacy implementations, and legacy histories replay unchanged. This test normalises
the import lines and a few path strings, then compares the rest of each file with its source. Sources are looked up
under $AGENTIC_ESOPT (default ~/Project/ZO/Agentic-ESOpt), $AGENTES_SYSTEM and $ES_SEAGENT; a missing source is
skipped, so the test runs wherever the legacy checkouts exist.
"""
from __future__ import annotations

import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home() / "Project"
AESO = Path(os.environ.get("AGENTIC_ESOPT", HOME / "ZO" / "Agentic-ESOpt"))
AGSYS = Path(os.environ.get("AGENTES_SYSTEM", HOME / "AgentES-System"))
SEA = Path(os.environ.get("ES_SEAGENT", HOME / "ES-SEAgent"))

BACKENDS = ["seeded_model_es", "structured_es", "batched_es", "dense_inplace", "fura_native", "fura_s_pretransform",
            "fura_s_quant", "lora_slots", "rot_hf", "rot_slots", "rot_tp", "steer_slots"]
PAIRS = [(f"esx/backends/{m}.py", AESO / "algorithms/es" / f"{m}.py") for m in BACKENDS] + [
    ("esx/run_state.py", AESO / "algorithms/es/run_state.py"),
    ("esx/registry.py", AESO / "algorithms/es/registry.py"),
    ("esx/serving/slot_engine.py", AESO / "algorithms/es/fura_vllm_engine.py"),
    ("esx/serving/http_client.py", AESO / "algorithms/es/model_es_client.py"),
    ("esx/serving/collective_client.py", AESO / "mobileworld-train-time/mwt/es_client.py"),
    ("esx/backends/mwt/param_es_worker.py", AESO / "scripts/motivation/param_es_worker.py"),
    ("esx/backends/mwt/fura_worker.py", AESO / "mobileworld-train-time/workers/fura_worker.py"),
    ("esx/backends/mwt/fura_blocks.py", AESO / "mobileworld-train-time/mwt/fura.py"),
    ("esx/backends/flipout_dense.py", AGSYS / "methods/flipout_dense.py"),
    ("esx/serving/flipout_engine.py", AGSYS / "methods/engine.py"),
    ("esx/estimators/entropic.py", SEA / "methods/discovery/entropic.py"),
    ("esx/estimators/archive.py", SEA / "methods/discovery/archive.py"),
]

_RENAMES = [  # specific rules first, then the generic module-path rules
    (r"from esx\.backends\.mwt import param_es_worker as pew", "import param_es_worker as pew"),
    (r"from (mwt\.fura|esx\.backends\.mwt\.fura_blocks) import", "from FURA import"),
    (r"from (methods\.discovery|esx\.estimators)\.entropic import", "from ENT import"),
    (r"from (methods|esx\.backends)\.flipout_dense import", "from FLIP import"),
    (r'"(esx\.backends\.mwt\.)?param_es_worker\.WorkerExtension"', '"PEW_EXT"'),
    (r"algorithms\.es\.fura_vllm_engine", "MOD"), (r"esx\.serving\.slot_engine", "MOD"),
    (r"algorithms\.es\.model_es_client", "MOD"), (r"esx\.serving\.http_client", "MOD"),
    (r"algorithms\.es\.(run_state|registry)", "MOD"), (r"esx\.(run_state|registry)", "MOD"),
    (r"algorithms\.es\.\w+", "MOD"), (r"esx\.backends\.\w+", "MOD"),
]


def _norm(text: str) -> list:
    out = []
    for line in text.splitlines():
        if "sys.path.insert" in line:  # repository-root hacks differ by construction
            continue
        for a, b in _RENAMES:
            line = re.sub(a, b, line)
        out.append(line.rstrip())
    return out


class LegacyParityTest(unittest.TestCase):
    def test_moved_files_equal_their_sources(self) -> None:
        checked = 0
        for new, old in PAIRS:
            with self.subTest(new=new):
                if not old.exists():
                    self.skipTest(f"legacy source not present: {old}")
                a, b = _norm((ROOT / new).read_text()), _norm(old.read_text())
                self.maxDiff = 2000
                self.assertEqual(a, b, f"{new} differs from {old} beyond import paths")
                checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
