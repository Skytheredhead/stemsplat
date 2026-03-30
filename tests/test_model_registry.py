from __future__ import annotations

import ast
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "main.py"
DOWNLOADER_PATH = ROOT / "downloader.py"
CONFIG_DIR = ROOT / "configs"

NEW_MODEL_EXPECTATIONS = {
    "guitar": {
        "config": "config_guitar_becruily.yaml",
        "segment": 485_100,
        "overlap": 2,
        "url": "https://huggingface.co/becruily/mel-band-roformer-guitar/resolve/main/becruily_guitar.ckpt?download=true",
        "filename": "becruily_guitar.ckpt",
    },
    "mel_band_karaoke": {
        "config": "config_karaoke_becruily.yaml",
        "segment": 485_100,
        "overlap": 8,
        "url": "https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/mel_band_roformer_karaoke_becruily.ckpt?download=true",
        "filename": "mel_band_roformer_karaoke_becruily.ckpt",
    },
    "denoise": {
        "config": "model_mel_band_roformer_denoise.yaml",
        "segment": 352_800,
        "overlap": 4,
        "url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt?download=true",
        "filename": "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
    },
    "bs_roformer_6s": {
        "config": "BS-Rofo-SW-Fixed.yaml",
        "segment": 588_800,
        "overlap": 2,
        "url": "https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt?download=true",
        "filename": "BS-Rofo-SW-Fixed.ckpt",
    },
    "htdemucs_ft_drums": {
        "config": "config_musdb18_htdemucs.yaml",
        "segment": 485_100,
        "overlap": 4,
        "url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th",
        "filename": "f7e0c4bc-ba3fe64a.th",
    },
    "htdemucs_ft_bass": {
        "config": "config_musdb18_htdemucs.yaml",
        "segment": 485_100,
        "overlap": 4,
        "url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/d12395a8-e57c48e6.th",
        "filename": "d12395a8-e57c48e6.th",
    },
    "htdemucs_ft_other": {
        "config": "config_musdb18_htdemucs.yaml",
        "segment": 485_100,
        "overlap": 4,
        "url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/92cfc3b6-ef3bcb9c.th",
        "filename": "92cfc3b6-ef3bcb9c.th",
    },
    "htdemucs_ft_vocals": {
        "config": "config_musdb18_htdemucs.yaml",
        "segment": 485_100,
        "overlap": 4,
        "url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th",
        "filename": "04573f0d-f3cf25b2.th",
    },
    "htdemucs_6s": {
        "config": "config_htdemucs_6stems.yaml",
        "segment": 485_100,
        "overlap": 4,
        "url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th",
        "filename": "5c90dfd2-34c22ccb.th",
    },
    "drumsep_6s": {
        "config": "aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml",
        "segment": 130_560,
        "overlap": 4,
        "url": "https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt",
        "filename": "aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt",
    },
    "drumsep_4s": {
        "config": "config_drumsep.yaml",
        "segment": 1_764_000,
        "overlap": 4,
        "url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.5/model_drumsep.th",
        "filename": "model_drumsep.th",
    },
}

PRESET_MODE_EXPECTATIONS = {
    "preset_voc_instrum": {
        "stems": ["voc_instrum"],
        "required_models": ["vocals", "instrumental"],
        "output_labels": ["vocals", "instrumental"],
    },
    "preset_boost_harmonies": {
        "stems": ["boost_harmonies"],
        "required_models": ["vocals", "mel_band_karaoke"],
        "output_labels": ["boost harmonies"],
    },
    "preset_boost_guitar": {
        "stems": ["boost_guitar"],
        "required_models": ["guitar"],
        "output_labels": ["boost guitar"],
    },
}


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _assignment_value(module: ast.Module, name: str) -> ast.AST:
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            if node.value is None:
                break
            return node.value
    raise AssertionError(f"assignment {name} not found")


def _literal_eval(module: ast.Module, name: str):
    return ast.literal_eval(_assignment_value(module, name))


def _simple_constants(module: ast.Module) -> dict[str, object]:
    values: dict[str, object] = {}
    deferred: list[tuple[str, ast.AST]] = []
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except Exception:
                    deferred.append((target.id, node.value))
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                values[node.target.id] = ast.literal_eval(node.value)
            except Exception:
                deferred.append((node.target.id, node.value))
    unresolved = list(deferred)
    while unresolved:
        next_pass: list[tuple[str, ast.AST]] = []
        progress = False
        for name, value_node in unresolved:
            if isinstance(value_node, ast.Name) and value_node.id in values:
                values[name] = values[value_node.id]
                progress = True
            else:
                next_pass.append((name, value_node))
        if not progress:
            break
        unresolved = next_pass
    return values


def _dict_with_named_constants(module: ast.Module, name: str, constants: dict[str, object]) -> dict[str, object]:
    node = _assignment_value(module, name)
    if not isinstance(node, ast.Dict):
        raise AssertionError(f"{name} is not a dict literal")
    payload: dict[str, object] = {}
    for key_node, value_node in zip(node.keys, node.values, strict=False):
        key = ast.literal_eval(key_node)
        try:
            payload[str(key)] = ast.literal_eval(value_node)
            continue
        except Exception:
            pass
        if isinstance(value_node, ast.Name) and value_node.id in constants:
            payload[str(key)] = constants[value_node.id]
            continue
        continue
    return payload


def _model_specs_from_ast(module: ast.Module) -> dict[str, dict[str, object]]:
    node = _assignment_value(module, "MODEL_SPECS")
    if not isinstance(node, ast.Dict):
        raise AssertionError("MODEL_SPECS is not a dict literal")
    specs: dict[str, dict[str, object]] = {}
    for key_node, value_node in zip(node.keys, node.values, strict=False):
        key = ast.literal_eval(key_node)
        if not isinstance(value_node, ast.Call):
            raise AssertionError(f"MODEL_SPECS[{key}] is not a ModelSpec call")
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in value_node.keywords if kw.arg}
        specs[str(key)] = kwargs
    return specs


class ModelRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.main_ast = _module_ast(MAIN_PATH)
        cls.downloader_ast = _module_ast(DOWNLOADER_PATH)
        cls.constants = _simple_constants(cls.main_ast)
        cls.model_specs = _model_specs_from_ast(cls.main_ast)
        cls.mode_to_stems = _literal_eval(cls.main_ast, "MODE_TO_STEMS")
        cls.mode_required_models = _literal_eval(cls.main_ast, "MODE_REQUIRED_MODELS")
        cls.mode_output_labels = _literal_eval(cls.main_ast, "MODE_OUTPUT_LABELS")
        cls.compat_defaults = _dict_with_named_constants(cls.main_ast, "COMPAT_SETTINGS_DEFAULTS", cls.constants)
        cls.downloader_files = _literal_eval(cls.downloader_ast, "FILES")

    def test_new_model_specs_match_configs(self) -> None:
        for key, expected in NEW_MODEL_EXPECTATIONS.items():
            spec = self.model_specs[key]
            self.assertEqual(spec["config"], expected["config"])
            self.assertEqual(spec["segment"], expected["segment"])
            self.assertEqual(spec["overlap"], expected["overlap"])

            config_path = CONFIG_DIR / str(spec["config"])
            cfg = yaml.unsafe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(int(cfg["audio"]["chunk_size"]), spec["segment"])
            self.assertEqual(int(cfg["inference"]["num_overlap"]), spec["overlap"])

    def test_new_modes_round_trip(self) -> None:
        for mode in NEW_MODEL_EXPECTATIONS:
            if mode in self.mode_to_stems:
                self.assertEqual(list(self.mode_to_stems[mode]), [mode])
                self.assertEqual(list(self.mode_required_models[mode]), [mode])

        self.assertEqual(list(self.mode_to_stems["both_separate"]), ["vocals", "instrumental"])
        self.assertEqual(list(self.mode_required_models["both_separate"]), ["vocals", "instrumental"])
        self.assertEqual(list(self.mode_to_stems["denoise"]), ["denoise"])
        self.assertEqual(list(self.mode_required_models["denoise"]), ["denoise"])

    def test_preset_mode_registry(self) -> None:
        for mode, expected in PRESET_MODE_EXPECTATIONS.items():
            self.assertEqual(list(self.mode_to_stems[mode]), expected["stems"])
            self.assertEqual(list(self.mode_required_models[mode]), expected["required_models"])
            self.assertEqual(list(self.mode_output_labels[mode]), expected["output_labels"])

    def test_preset_defaults_are_present(self) -> None:
        self.assertEqual(self.compat_defaults["previous_files_retention"], "1w")
        self.assertEqual(self.compat_defaults["previous_files_limit_gb"], 10.0)
        self.assertEqual(self.compat_defaults["previous_files_warn_gb"], 8.0)
        self.assertEqual(self.compat_defaults["boost_harmonies_background_vocals_gain_db"], 3.0)
        self.assertEqual(self.compat_defaults["boost_harmonies_base_song_gain_db"], -3.0)
        self.assertEqual(self.compat_defaults["boost_guitar_guitar_gain_db"], 3.0)
        self.assertEqual(self.compat_defaults["boost_guitar_base_song_gain_db"], -3.0)

    def test_downloader_entries_match_model_specs(self) -> None:
        by_tag = {item["tag"]: item for item in self.downloader_files}
        for key, expected in NEW_MODEL_EXPECTATIONS.items():
            item = by_tag[key]
            self.assertEqual(item["filename"], expected["filename"])
            self.assertEqual(item["url"], expected["url"])


if __name__ == "__main__":
    unittest.main()
