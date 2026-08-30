import pytest
from proximalfm._model.inference_config import MgrConfig, InferenceConfig


class TestMgrConfig:
    def test_mapping_interface(self):
        config = MgrConfig(use_amp=True, min_batch_size=2)
        assert {"use_amp", "min_batch_size"} <= set(config.keys())
        items = dict(config.items())
        assert items["use_amp"] is True
        assert items["min_batch_size"] == 2
        assert "use_amp" in list(config)
        assert config["use_amp"] is True
        with pytest.raises(KeyError):
            _ = config["invalid_key"]
        assert config.get("use_amp") is True
        assert config.get("verbose", default=False) is False
        with pytest.raises(KeyError):
            config.get("invalid_key")

    def test_update(self):
        config = MgrConfig(use_amp=True, min_batch_size=2)

        config.update({"use_amp": False, "safety_factor": 0.5})
        assert config.use_amp is False
        assert config.safety_factor == 0.5
        other_config = MgrConfig(min_batch_size=4)
        config.update(other_config)
        assert config.min_batch_size == 4
        with pytest.raises(TypeError):
            config.update(["not_a_dict"])

    def test_invalid_construction_is_rejected(self):
        cases = [
            ({"invalid_key": True}, KeyError, "Invalid config key"),
            ({"use_amp": "not_a_bool"}, TypeError, "must be a boolean"),
            ({"min_batch_size": 0}, ValueError, "must be an integer"),
            ({"safety_factor": 1.5}, ValueError, "must be a float"),
        ]
        for kwargs, error, message in cases:
            with pytest.raises(error, match=message):
                MgrConfig(**kwargs)


class TestInferenceConfig:
    def test_defaults_and_overrides(self):
        config = InferenceConfig()
        assert isinstance(config.COL_CONFIG, MgrConfig)
        assert isinstance(config.ROW_CONFIG, MgrConfig)
        assert isinstance(config.ICL_CONFIG, MgrConfig)
        assert config.COL_CONFIG.use_amp is True
        assert config.COL_CONFIG.min_batch_size == 1
        assert config.COL_CONFIG.offload == "auto"

        config = InferenceConfig(
            COL_CONFIG={"use_amp": False, "min_batch_size": 2},
            ROW_CONFIG={"offload": "cpu"},
        )
        assert config.COL_CONFIG.use_amp is False
        assert config.COL_CONFIG.min_batch_size == 2
        assert config.ROW_CONFIG.offload == "cpu"
        assert config.ICL_CONFIG.use_amp is True

        config = InferenceConfig(COL_CONFIG=MgrConfig(use_amp=False))
        assert config.COL_CONFIG.use_amp is False

    def test_invalid_section_types_are_rejected(self):
        for section in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG"):
            with pytest.raises(TypeError, match="must be a dict or MgrConfig"):
                InferenceConfig(**{section: ["invalid"]})

    def test_update_from_dict_valid(self):
        config = InferenceConfig()
        update_dict = {
            "COL_CONFIG": {"use_amp": False, "min_batch_size": 4},
            "ROW_CONFIG": {"offload": "cpu"},
        }
        config.update_from_dict(update_dict)

        assert config.COL_CONFIG.use_amp is False
        assert config.COL_CONFIG.min_batch_size == 4
        assert config.ROW_CONFIG.offload == "cpu"

    def test_update_from_dict_rejects_invalid_keys(self):
        config = InferenceConfig()
        with pytest.raises(KeyError, match="Invalid config key"):
            config.update_from_dict({"INVALID_CONFIG": {"use_amp": False}})
        with pytest.raises(KeyError, match="Invalid config key"):
            config.update_from_dict({"COL_CONFIG": {"invalid_key": False}})
