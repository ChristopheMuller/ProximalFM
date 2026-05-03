import pytest
import proximalfm._model.attention as attention
from proximalfm._model.attention import flash_attn3_toggle

def test_flash_attn3_toggle_enables():
    original_value = attention._use_flash_attn3

    with flash_attn3_toggle(True):
        assert attention._use_flash_attn3 is True

    assert attention._use_flash_attn3 == original_value

def test_flash_attn3_toggle_disables():
    original_value = attention._use_flash_attn3

    # Set to true initially to test disabling
    attention._use_flash_attn3 = True

    with flash_attn3_toggle(False):
        assert attention._use_flash_attn3 is False

    assert attention._use_flash_attn3 is True

    # Restore original value
    attention._use_flash_attn3 = original_value

def test_flash_attn3_toggle_exception_restores():
    original_value = attention._use_flash_attn3

    with pytest.raises(ValueError):
        with flash_attn3_toggle(not original_value):
            assert attention._use_flash_attn3 == (not original_value)
            raise ValueError("Test exception")

    assert attention._use_flash_attn3 == original_value
