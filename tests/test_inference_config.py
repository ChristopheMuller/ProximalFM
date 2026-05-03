import pytest
import torch
from proximalfm._model.inference_config import MgrConfig, InferenceConfig

class TestMgrConfig:
    def test_init_valid(self):
        config = MgrConfig(use_amp=True, min_batch_size=2)
        assert config.use_amp is True
        assert config.min_batch_size == 2

    def test_init_invalid_key(self):
        with pytest.raises(KeyError, match="Invalid config key"):
            MgrConfig(invalid_key=True)

    def test_init_invalid_type(self):
        with pytest.raises(TypeError, match="must be a boolean"):
            MgrConfig(use_amp="not_a_bool")

    def test_init_invalid_value(self):
        with pytest.raises(ValueError, match="must be an integer"):
            MgrConfig(min_batch_size=0)

        with pytest.raises(ValueError, match="must be a float"):
            MgrConfig(safety_factor=1.5)

    def test_methods(self):
        config = MgrConfig(use_amp=True, min_batch_size=2)

        # Test keys
        keys = config.keys()
        assert "use_amp" in keys
        assert "min_batch_size" in keys

        # Test items
        items = dict(config.items())
        assert items["use_amp"] is True
        assert items["min_batch_size"] == 2

        # Test __iter__
        assert "use_amp" in list(config)

        # Test __getitem__
        assert config["use_amp"] is True

        # Test __getitem__ with invalid key
        with pytest.raises(KeyError):
            _ = config["invalid_key"]

        # Test get
        assert config.get("use_amp") is True
        assert config.get("verbose", default=False) is False

        # Test get with invalid key
        with pytest.raises(KeyError):
            config.get("invalid_key")

    def test_update(self):
        config = MgrConfig(use_amp=True, min_batch_size=2)

        # Update with dict
        config.update({"use_amp": False, "safety_factor": 0.5})
        assert config.use_amp is False
        assert config.safety_factor == 0.5

        # Update with another MgrConfig
        other_config = MgrConfig(min_batch_size=4)
        config.update(other_config)
        assert config.min_batch_size == 4

        # Update with invalid type
        with pytest.raises(TypeError):
            config.update(["not_a_dict"])


class TestInferenceConfig:
    def test_init_valid_defaults(self):
        config = InferenceConfig()
        assert isinstance(config.COL_CONFIG, MgrConfig)
        assert isinstance(config.ROW_CONFIG, MgrConfig)
        assert isinstance(config.ICL_CONFIG, MgrConfig)

        # Check some defaults
        assert config.COL_CONFIG.use_amp is True
        assert config.COL_CONFIG.min_batch_size == 1
        assert config.COL_CONFIG.offload == "auto"

    def test_init_with_dicts(self):
        config = InferenceConfig(
            COL_CONFIG={"use_amp": False, "min_batch_size": 2},
            ROW_CONFIG={"offload": "cpu"}
        )
        assert config.COL_CONFIG.use_amp is False
        assert config.COL_CONFIG.min_batch_size == 2
        assert config.ROW_CONFIG.offload == "cpu"

        # Check ICL_CONFIG is still default MgrConfig
        assert isinstance(config.ICL_CONFIG, MgrConfig)
        assert config.ICL_CONFIG.use_amp is True

    def test_init_with_mgrconfigs(self):
        col_mgr = MgrConfig(use_amp=False)
        config = InferenceConfig(COL_CONFIG=col_mgr)
        assert config.COL_CONFIG.use_amp is False

    def test_init_invalid_col_config_type(self):
        with pytest.raises(TypeError, match="must be a dict or MgrConfig"):
            InferenceConfig(COL_CONFIG=["invalid"])

    def test_init_invalid_row_config_type(self):
        with pytest.raises(TypeError, match="must be a dict or MgrConfig"):
            InferenceConfig(ROW_CONFIG=["invalid"])

    def test_init_invalid_icl_config_type(self):
        with pytest.raises(TypeError, match="must be a dict or MgrConfig"):
            InferenceConfig(ICL_CONFIG=["invalid"])

    def test_update_from_dict_valid(self):
        config = InferenceConfig()
        update_dict = {
            "COL_CONFIG": {"use_amp": False, "min_batch_size": 4},
            "ROW_CONFIG": {"offload": "cpu"}
        }
        config.update_from_dict(update_dict)

        assert config.COL_CONFIG.use_amp is False
        assert config.COL_CONFIG.min_batch_size == 4
        assert config.ROW_CONFIG.offload == "cpu"

    def test_update_from_dict_invalid_key(self):
        config = InferenceConfig()
        update_dict = {
            "INVALID_CONFIG": {"use_amp": False}
        }
        with pytest.raises(KeyError, match="Invalid config key"):
            config.update_from_dict(update_dict)

    def test_update_from_dict_invalid_mgr_key(self):
        config = InferenceConfig()
        update_dict = {
            "COL_CONFIG": {"invalid_key": False}
        }
        with pytest.raises(KeyError, match="Invalid config key"):
            config.update_from_dict(update_dict)
