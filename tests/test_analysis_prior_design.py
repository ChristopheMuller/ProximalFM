"""Tests for analysis_prior.design (the shared visual identity)."""

import pytest

from analysis_prior import design


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------


def test_figure_size_helpers_preserve_paper_width_and_panel_aspect():
    width, height = design.figsize(fraction=1.0)
    assert width == pytest.approx(design.TEXT_WIDTH_IN)
    assert height == pytest.approx(design.TEXT_WIDTH_IN * design.GOLDEN)
    assert design.figsize(fraction=0.5)[0] == pytest.approx(design.TEXT_WIDTH_IN / 2)

    for ncols in (1, 3, 4):
        for nrows in (1, 3):
            width, height = design.figsize_grid(ncols=ncols, nrows=nrows)
            assert width == pytest.approx(design.TEXT_WIDTH_IN)
            expected = design.TEXT_WIDTH_IN / ncols * nrows
            assert height == pytest.approx(expected)

    _, h1 = design.figsize_grid(ncols=3, nrows=1)
    _, h2 = design.figsize_grid(ncols=3, nrows=2)
    assert h2 == pytest.approx(2 * h1)
    _, h_extra = design.figsize_grid(ncols=3, nrows=1, extra_height_in=0.5)
    assert h_extra == pytest.approx(h1 + 0.5)


# --------------------------------------------------------------------------
# Semantic palettes
# --------------------------------------------------------------------------


def test_block_palette_is_semantic_distinct_and_colorblind_safe():
    assert design.block_color("U") == design.block_color("d_U")
    colors = list(design.BLOCK_COLORS.values())
    assert len(set(colors)) == len(colors)
    assert design.block_color("U") == design.OKABE_ITO["vermillion"]
    safe = set(design.OKABE_ITO.values())
    for color in [
        *design.BLOCK_COLORS.values(),
        *design.VARIANT_COLORS.values(),
        *design._VARIANT_CYCLE,
    ]:
        try:
            float(color)  # achromatic matplotlib colour
        except (TypeError, ValueError):
            assert color in safe


def test_block_color_rejects_unknown_roles():
    with pytest.raises(KeyError, match="No colour for block"):
        design.block_color("Q")


# --------------------------------------------------------------------------
# Variant styles
# --------------------------------------------------------------------------


def test_categorical_variant_styles_are_stable_and_ordered():
    a = design.variant_styles(["triangular", "uniform"])
    b = design.variant_styles(["unenforced", "uniform", "triangular"])
    assert a["triangular"] == b["triangular"]
    assert a["uniform"] == b["uniform"]
    styles = design.variant_styles(["alpha", "beta"])
    assert styles["alpha"]["color"] == design._VARIANT_CYCLE[0]
    assert styles["beta"]["color"] == design._VARIANT_CYCLE[1]
    labels = ["triangular", "zzz", "uniform"]
    assert list(design.variant_styles(labels)) == labels


def test_ordered_variant_styles_use_a_sequential_ramp_and_handle_edges():
    labels = ["K1", "K5", "K25", "K200"]
    styles = design.variant_styles(labels, ordered=True)
    assert list(styles) == labels
    colors = [s["color"] for s in styles.values()]
    assert len(set(colors)) == len(colors)
    assert set(colors).isdisjoint(set(design.VARIANT_COLORS.values()))
    assert len(design.variant_styles(["K1"], ordered=True)) == 1
    assert design.variant_styles([]) == {}


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------


def test_apply_style_matches_the_paper_and_is_idempotent():
    design.apply_style()
    first = dict(design.plt.rcParams)
    design.apply_style()
    assert design.plt.rcParams["font.size"] == first["font.size"]
    assert design.plt.rcParams["figure.constrained_layout.use"] is True
    assert design.plt.rcParams["savefig.bbox"] is None
    assert design.plt.rcParams["font.size"] == design.FONT_SIZE_PT


# --------------------------------------------------------------------------
# Idioms
# --------------------------------------------------------------------------


def test_reference_line_helpers_apply_their_semantics():
    design.apply_style()
    fig, ax = design.plt.subplots()
    design.constraint_line(ax, 0, 5, label="$d_U = d_W$")

    (line,) = ax.get_lines()
    assert list(line.get_xdata()) == [0, 5]
    assert list(line.get_ydata()) == [0, 5]
    assert line.get_linestyle() == "--"
    assert ax.get_legend() is not None
    design.plt.close(fig)

    design.apply_style()
    fig, ax = design.plt.subplots()
    design.theory_line(ax, [1, 2], [0.1, 0.2])

    (line,) = ax.get_lines()
    assert line.get_linestyle() == "--"
    assert line.get_label() == "theory"
    design.plt.close(fig)


def test_grouped_bar_offsets_are_centred_and_non_overlapping():
    for n in (1, 2, 3, 4):
        offsets = design.grouped_bar_offsets(n)
        assert len(offsets) == n
        assert sum(offsets) == pytest.approx(0.0)
    n, total = 3, 0.8
    offsets = design.grouped_bar_offsets(n, total_width=total)
    width = total / n
    gaps = [b - a for a, b in zip(offsets, offsets[1:])]
    assert all(gap == pytest.approx(width) for gap in gaps)
