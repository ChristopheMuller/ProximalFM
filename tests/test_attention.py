import pytest
import proximalfm._model.attention as attention
from proximalfm._model.attention import flash_attn3_toggle


def test_flash_attn3_toggle_sets_and_restores_state():
    original_value = attention._use_flash_attn3
    for enabled in (True, False):
        with flash_attn3_toggle(enabled):
            assert attention._use_flash_attn3 is enabled
        assert attention._use_flash_attn3 == original_value

    with pytest.raises(ValueError):
        with flash_attn3_toggle(not original_value):
            assert attention._use_flash_attn3 == (not original_value)
            raise ValueError("Test exception")

    assert attention._use_flash_attn3 == original_value
