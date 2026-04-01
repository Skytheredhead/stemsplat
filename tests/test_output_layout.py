from __future__ import annotations

import ast
import contextlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "main.py"


def _load_helpers():
    module = ast.parse(MAIN_PATH.read_text(encoding="utf-8"), filename=str(MAIN_PATH))
    wanted_assignments = {"ALL_STEMS_DRUM_OUTPUT_LABELS"}
    wanted_functions = {"_output_subdir_for_label", "_relative_output_name"}
    selected: list[ast.AST] = []
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted_assignments:
                    selected.append(node)
                    break
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            selected.append(node)

    helper_module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(helper_module)
    namespace = {"Path": Path, "contextlib": contextlib}
    exec(compile(helper_module, str(MAIN_PATH), "exec"), namespace)
    return namespace


class OutputLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        namespace = _load_helpers()
        cls.output_subdir_for_label = staticmethod(namespace["_output_subdir_for_label"])
        cls.relative_output_name = staticmethod(namespace["_relative_output_name"])

    def test_all_stems_drum_outputs_are_grouped_under_drums_folder(self) -> None:
        for label in ("kick", "snare", "toms", "hh", "ride", "crash"):
            self.assertEqual(self.output_subdir_for_label("preset_all_stems", label), Path("drums"))

    def test_non_drum_outputs_stay_at_top_level(self) -> None:
        self.assertIsNone(self.output_subdir_for_label("preset_all_stems", "vocals"))
        self.assertIsNone(self.output_subdir_for_label("drumsep_6s", "kick"))

    def test_relative_output_name_preserves_nested_paths(self) -> None:
        root = Path("/tmp/stemsplat-output")
        nested = root / "drums" / "song - kick.wav"
        self.assertEqual(self.relative_output_name(nested, root), "drums/song - kick.wav")


if __name__ == "__main__":
    unittest.main()
