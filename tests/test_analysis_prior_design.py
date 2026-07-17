"""Tests for analysis_prior.design (the shared visual identity)."""

import pytest

from analysis_prior import design


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------


def test_figsize_spans_the_text_width():
    # Figures are included unscaled, so a full-width figure must be exactly the
    # text width -- otherwise LaTeX rescales it and the font size drifts.
    width, height = design.figsize(fraction=1.0)
    assert width == pytest.approx(design.TEXT_WIDTH_IN)
    assert height == pytest.approx(design.TEXT_WIDTH_IN * design.GOLDEN)


def test_figsize_fraction():
    assert design.figsize(fraction=0.5)[0] == pytest.approx(design.TEXT_WIDTH_IN / 2)


def test_figsize_grid_never_exceeds_the_text_width():
    for ncols in (1, 3, 4):
        for nrows in (1, 3):
            width, _ = design.figsize_grid(ncols=ncols, nrows=nrows)
            assert width == pytest.approx(design.TEXT_WIDTH_IN)


def test_figsize_grid_panels_keep_their_aspect_as_columns_grow():
    # Height follows from panel width, so panels stay square-ish instead of
    # squashing when columns are added.
    _, h3 = design.figsize_grid(ncols=3, panel_aspect=1.0)
    _, h6 = design.figsize_grid(ncols=6, panel_aspect=1.0)
    assert h3 == pytest.approx(design.TEXT_WIDTH_IN / 3)
    assert h6 == pytest.approx(design.TEXT_WIDTH_IN / 6)


def test_figsize_grid_rows_and_extra_height():
    _, h1 = design.figsize_grid(ncols=3, nrows=1)
    _, h2 = design.figsize_grid(ncols=3, nrows=2)
    assert h2 == pytest.approx(2 * h1)
    _, h_extra = design.figsize_grid(ncols=3, nrows=1, extra_height_in=0.5)
    assert h_extra == pytest.approx(h1 + 0.5)


# --------------------------------------------------------------------------
# Semantic palettes
# --------------------------------------------------------------------------


def test_block_color_accepts_both_spellings():
    assert design.block_color("U") == design.block_color("d_U")


def test_block_colors_are_distinct():
    colors = list(design.BLOCK_COLORS.values())
    assert len(set(colors)) == len(colors)


def test_latent_confounder_is_red_like_the_dag():
    # The main-text DAG draws the latent node in red; the plots must agree, so a
    # reader only learns "red = latent" once.
    assert design.block_color("U") == design.OKABE_ITO["vermillion"]


def test_block_color_unknown_raises_with_a_useful_message():
    with pytest.raises(KeyError, match="No colour for block"):
        design.block_color("Q")


def test_palettes_are_colorblind_safe_okabe_ito():
    # Everything categorical must come from the Okabe-Ito set.
    safe = set(design.OKABE_ITO.values())
    assert set(design.BLOCK_COLORS.values()) <= safe
    assert set(design.VARIANT_COLORS.values()) <= safe
    assert set(design._VARIANT_CYCLE) <= safe


# --------------------------------------------------------------------------
# Variant styles
# --------------------------------------------------------------------------


def test_variant_styles_pin_known_labels_across_families():
    # The whole point: 'triangular' is the same colour in every figure, whatever
    # else it is shown next to.
    a = design.variant_styles(["triangular", "uniform"])
    b = design.variant_styles(["unenforced", "uniform", "triangular"])
    assert a["triangular"] == b["triangular"]
    assert a["uniform"] == b["uniform"]


def test_variant_styles_assign_unknown_labels_in_order():
    styles = design.variant_styles(["alpha", "beta"])
    assert styles["alpha"]["color"] == design._VARIANT_CYCLE[0]
    assert styles["beta"]["color"] == design._VARIANT_CYCLE[1]


def test_variant_styles_preserve_label_order():
    labels = ["triangular", "zzz", "uniform"]
    assert list(design.variant_styles(labels)) == labels


def test_variant_styles_ordered_uses_a_sequential_ramp():
    # A K-sweep is ordered; a categorical palette would discard that.
    labels = ["K1", "K5", "K25", "K200"]
    styles = design.variant_styles(labels, ordered=True)
    assert list(styles) == labels
    colors = [s["color"] for s in styles.values()]
    assert len(set(colors)) == len(colors)
    # Ordered mode ignores the pinned categorical colours entirely.
    assert set(colors).isdisjoint(set(design.VARIANT_COLORS.values()))


def test_variant_styles_ordered_handles_a_single_label():
    assert len(design.variant_styles(["K1"], ordered=True)) == 1


def test_variant_styles_empty():
    assert design.variant_styles([]) == {}


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------


def test_apply_style_is_idempotent():
    design.apply_style()
    first = dict(design.plt.rcParams)
    design.apply_style()
    assert design.plt.rcParams["font.size"] == first["font.size"]


def test_apply_style_uses_constrained_layout_not_tight_bbox():
    # bbox="tight" crops after the fact, changing the saved width and breaking
    # the unscaled-inclusion scheme; constrained layout fits within the size.
    design.apply_style()
    assert design.plt.rcParams["figure.constrained_layout.use"] is True
    assert design.plt.rcParams["savefig.bbox"] is None


def test_font_size_matches_the_paper():
    # Figures are included unscaled, so this is literally the rendered size.
    design.apply_style()
    assert design.plt.rcParams["font.size"] == design.FONT_SIZE_PT


# --------------------------------------------------------------------------
# Idioms
# --------------------------------------------------------------------------


def test_constraint_line_draws_the_identity_and_labels_it():
    design.apply_style()
    fig, ax = design.plt.subplots()
    design.constraint_line(ax, 0, 5, label="$d_U = d_W$")

    (line,) = ax.get_lines()
    assert list(line.get_xdata()) == [0, 5]
    assert list(line.get_ydata()) == [0, 5]
    assert line.get_linestyle() == "--"
    assert ax.get_legend() is not None
    design.plt.close(fig)


def test_theory_line_uses_the_theory_role():
    design.apply_style()
    fig, ax = design.plt.subplots()
    design.theory_line(ax, [1, 2], [0.1, 0.2])

    (line,) = ax.get_lines()
    assert line.get_linestyle() == "--"
    assert line.get_label() == "theory"
    design.plt.close(fig)


def test_grouped_bar_offsets_are_centred():
    for n in (1, 2, 3, 4):
        offsets = design.grouped_bar_offsets(n)
        assert len(offsets) == n
        assert sum(offsets) == pytest.approx(0.0)


def test_grouped_bar_offsets_do_not_overlap():
    n, total = 3, 0.8
    offsets = design.grouped_bar_offsets(n, total_width=total)
    width = total / n
    gaps = [b - a for a, b in zip(offsets, offsets[1:])]
    assert all(gap == pytest.approx(width) for gap in gaps)
