"""Unit tests for category_encoders.migration."""

import category_encoders as ce
import numpy as np
import pandas as pd
import pytest
from category_encoders.migration import (
    SumMigrationConflictError,
    _values_equal,
    migrate_onehot_pair_to_sum,
)
from sklearn.preprocessing import OneHotEncoder


def make_ohe(values, **kwargs):
    """Fit a sklearn OneHotEncoder on a single-column object array."""
    kwargs.setdefault('sparse_output', False)
    return OneHotEncoder(**kwargs).fit(np.array([[v] for v in values], dtype=object))


def old_scores(df, fa, fb, ohe_a, ohe_b, intercept, w_a, w_b, interaction):
    """Reference old linear score for a frame of raw values."""

    def encode(ohe, values):
        if getattr(ohe, 'feature_names_in_', None) is not None:
            data = pd.DataFrame({ohe.feature_names_in_[0]: np.asarray(values, dtype=object)})
        else:
            data = np.array(list(values), dtype=object).reshape(-1, 1)
        encoded = ohe.transform(data)
        if hasattr(encoded, 'todense'):
            encoded = encoded.todense()
        return np.asarray(encoded, dtype=float)

    z_a = encode(ohe_a, df[fa])
    z_b = encode(ohe_b, df[fb])
    return (
        intercept
        + z_a @ np.asarray(w_a, dtype=float)
        + z_b @ np.asarray(w_b, dtype=float)
        + np.einsum('ij,jk,ik->i', z_a, np.asarray(interaction, dtype=float), z_b)
    )


@pytest.fixture
def rng():
    """Deterministic random generator for coefficients."""
    return np.random.default_rng(20261120)


# ---------------------------------------------------------------------------
# exactness on the declared grid
# ---------------------------------------------------------------------------


def test_declared_grid_is_exact_and_score_matches_old(rng):
    """Declared combinations migrate exactly; score() reproduces the old score."""
    ohe_a = make_ohe(['a', 'b', 'c'], handle_unknown='ignore')
    ohe_b = make_ohe(['x', 'y'], handle_unknown='ignore')
    w_a, w_b = rng.normal(size=3), rng.normal(size=2)
    interaction = rng.normal(size=(3, 2))
    result = migrate_onehot_pair_to_sum(
        ohe_a,
        ohe_b,
        intercept=0.7,
        main_effects_a=w_a,
        main_effects_b=w_b,
        interaction=interaction,
        field_a='fa',
        field_b='fb',
        on_conflict='return',
    )
    assert result.report.max_abs_diff_declared < 1e-9
    assert result.report.n_declared_cells == 6
    # random coefficients are not zero-sum, so unseen/missing probes conflict
    assert not result.ok
    assert {c.kind for c in result.conflicts} <= {'missing', 'unseen'}

    df = pd.DataFrame({'fa': ['a', 'b', 'c', 'c', 'a'], 'fb': ['x', 'y', 'x', 'y', 'x']})
    expected = old_scores(df, 'fa', 'fb', ohe_a, ohe_b, 0.7, w_a, w_b, interaction)
    np.testing.assert_allclose(result.score(df), expected, rtol=1e-9, atol=1e-12)


def test_full_equivalence_including_unseen_and_missing():
    """Zero-sum old effects migrate exactly, including unseen/missing inputs."""
    ohe_a = make_ohe(['a', 'b', 'c'], handle_unknown='ignore')
    ohe_b = make_ohe(['x', 'y'], handle_unknown='ignore')
    w_a = np.array([1.0, -2.0, 1.0])
    w_b = np.array([0.5, -0.5])
    interaction = np.array([[0.2, -0.2], [-0.5, 0.5], [0.3, -0.3]])
    result = migrate_onehot_pair_to_sum(
        ohe_a,
        ohe_b,
        intercept=0.7,
        main_effects_a=w_a,
        main_effects_b=w_b,
        interaction=interaction,
        field_a='fa',
        field_b='fb',
    )
    assert result.ok
    assert result.report.max_abs_diff < 1e-9

    df = pd.DataFrame({'fa': ['a', 'never-seen', np.nan, 'b'], 'fb': ['x', 'y', 'y', 'never-seen']})
    expected = old_scores(df, 'fa', 'fb', ohe_a, ohe_b, 0.7, w_a, w_b, interaction)
    np.testing.assert_allclose(result.score(df), expected, rtol=1e-9, atol=1e-12)


def test_new_intercept_is_grand_mean_of_old_grid(rng):
    """The migrated intercept is the grand mean of the old declared-grid scores."""
    ohe_a = make_ohe(['a', 'b'], handle_unknown='ignore')
    ohe_b = make_ohe(['x', 'y', 'z'], handle_unknown='ignore')
    w_a, w_b = rng.normal(size=2), rng.normal(size=3)
    interaction = rng.normal(size=(2, 3))
    result = migrate_onehot_pair_to_sum(
        ohe_a,
        ohe_b,
        intercept=1.25,
        main_effects_a=w_a,
        main_effects_b=w_b,
        interaction=interaction,
        field_a='fa',
        field_b='fb',
        on_conflict='return',
    )
    grid = pd.DataFrame([(a, b) for a in ['a', 'b'] for b in ['x', 'y', 'z']], columns=['fa', 'fb'])
    old = old_scores(grid, 'fa', 'fb', ohe_a, ohe_b, 1.25, w_a, w_b, interaction)
    assert result.intercept == pytest.approx(old.mean(), abs=1e-12)


# ---------------------------------------------------------------------------
# conflicts
# ---------------------------------------------------------------------------


def test_unseen_conflict_is_specific(rng):
    """Unseen-input conflicts name the field, value, both encodings and scores."""
    ohe_a = make_ohe(['a', 'b', 'c'], handle_unknown='ignore')
    ohe_b = make_ohe(['x', 'y'], handle_unknown='ignore')
    w_a, w_b = rng.normal(size=3), rng.normal(size=2)
    interaction = rng.normal(size=(3, 2))
    result = migrate_onehot_pair_to_sum(
        ohe_a,
        ohe_b,
        intercept=0.0,
        main_effects_a=w_a,
        main_effects_b=w_b,
        interaction=interaction,
        field_a='fa',
        field_b='fb',
        on_conflict='return',
    )
    unseen = [c for c in result.conflicts if c.kind == 'unseen']
    assert unseen
    conflict = unseen[0]
    assert conflict.field_a == 'fa' and conflict.field_b == 'fb'
    assert 'unseen' in conflict.reason
    assert conflict.abs_diff == pytest.approx(
        abs(conflict.old_score - conflict.new_score), rel=1e-9
    )
    # the new score of an unseen field-A value is the column average of the
    # declared field-A scores (all-zero sum-contrast row)
    target = [c for c in unseen if c.value_a == '__UNSEEN_CATEGORY__' and c.value_b == 'x']
    assert len(target) == 1
    declared_means = np.mean(
        [
            old_scores(
                pd.DataFrame({'fa': [a], 'fb': ['x']}),
                'fa',
                'fb',
                ohe_a,
                ohe_b,
                0.0,
                w_a,
                w_b,
                interaction,
            )[0]
            for a in ['a', 'b', 'c']
        ]
    )
    assert target[0].new_score == pytest.approx(declared_means, rel=1e-9)


def test_raise_lists_conflicts(rng):
    """The default on_conflict='raise' surfaces the conflicts in the exception."""
    ohe_a = make_ohe(['a', 'b'], handle_unknown='ignore')
    ohe_b = make_ohe(['x', 'y'], handle_unknown='ignore')
    with pytest.raises(SumMigrationConflictError) as excinfo:
        migrate_onehot_pair_to_sum(
            ohe_a,
            ohe_b,
            intercept=0.0,
            main_effects_a=rng.normal(size=2),
            main_effects_b=rng.normal(size=2),
            interaction=rng.normal(size=(2, 2)),
            field_a='fa',
            field_b='fb',
        )
    message = str(excinfo.value)
    assert 'fa' in message
    assert excinfo.value.conflicts


def test_old_handle_unknown_error_makes_probes_out_of_scope(rng):
    """Probes the old encoder rejects are reported as notes, not conflicts."""
    ohe_a = make_ohe(['a', 'b', 'c'])  # handle_unknown='error' by default
    ohe_b = make_ohe(['x', 'y'])
    result = migrate_onehot_pair_to_sum(
        ohe_a,
        ohe_b,
        intercept=0.3,
        main_effects_a=rng.normal(size=3),
        main_effects_b=rng.normal(size=2),
        interaction=rng.normal(size=(3, 2)),
        field_a='fa',
        field_b='fb',
    )
    assert result.ok
    assert result.report.n_unscorable_cells > 0
    assert any('cannot score' in note for note in result.report.notes)


def test_undeclared_missing_conflicts_with_kind_missing(rng):
    """Undeclared missing values surface as 'missing' conflicts."""
    ohe_a = make_ohe(['a', 'b'], handle_unknown='ignore')
    ohe_b = make_ohe(['x', 'y'], handle_unknown='ignore')
    result = migrate_onehot_pair_to_sum(
        ohe_a,
        ohe_b,
        intercept=0.0,
        main_effects_a=rng.normal(size=2),
        main_effects_b=rng.normal(size=2),
        interaction=rng.normal(size=(2, 2)),
        field_a='fa',
        field_b='fb',
        on_conflict='return',
    )
    missing = [c for c in result.conflicts if c.kind == 'missing']
    assert missing
    assert any(pd.isna(c.value_a) or pd.isna(c.value_b) for c in missing)


def test_user_supplied_unseen_probe_is_checked(rng):
    """User-supplied unseen probe values are verified and reported."""
    ohe_a = make_ohe(['a', 'b'], handle_unknown='ignore')
    ohe_b = make_ohe(['x', 'y'], handle_unknown='ignore')
    result = migrate_onehot_pair_to_sum(
        ohe_a,
        ohe_b,
        intercept=0.0,
        main_effects_a=rng.normal(size=2),
        main_effects_b=rng.normal(size=2),
        interaction=rng.normal(size=(2, 2)),
        field_a='fa',
        field_b='fb',
        unseen_values_a=['weird-value'],
        on_conflict='return',
    )
    assert any(c.value_a == 'weird-value' for c in result.conflicts)


# ---------------------------------------------------------------------------
# declared missing values and None
# ---------------------------------------------------------------------------


def test_declared_missing_is_migrated(rng):
    """A missing value declared in the old encoder becomes a regular new level."""
    ohe_a = make_ohe(['a', 'b', np.nan], handle_unknown='ignore')
    ohe_b = make_ohe(['x', 'y'], handle_unknown='ignore')
    w_a, w_b = rng.normal(size=3), rng.normal(size=2)
    interaction = rng.normal(size=(3, 2))
    result = migrate_onehot_pair_to_sum(
        ohe_a,
        ohe_b,
        intercept=0.4,
        main_effects_a=w_a,
        main_effects_b=w_b,
        interaction=interaction,
        field_a='fa',
        field_b='fb',
        on_conflict='return',
    )
    assert result.report.max_abs_diff_declared < 1e-9
    assert any(pd.isna(v) for v in result.declared_categories_a)
    # nan is a regular declared level of the new encoder: a non-zero row
    encoded = result.encoder_a.transform(pd.DataFrame({'fa': [np.nan]}))
    assert encoded[result.feature_names_a].to_numpy().any()

    df = pd.DataFrame({'fa': ['a', np.nan, 'b', np.nan], 'fb': ['x', 'y', 'y', 'x']})
    expected = old_scores(df, 'fa', 'fb', ohe_a, ohe_b, 0.4, w_a, w_b, interaction)
    np.testing.assert_allclose(result.score(df), expected, rtol=1e-9, atol=1e-12)


def test_none_declared_in_old_encoder_is_kept_distinct(rng):
    """Declared None migrates exactly; undeclared nan maps to the None level."""
    # sklearn treats None as its own category; nan stays undeclared and the
    # SumEncoder merges nan into the None level, which must surface as a
    # 'missing' conflict.
    ohe_a = make_ohe(['a', 'b', None], handle_unknown='ignore')
    ohe_b = make_ohe(['x', 'y'], handle_unknown='ignore')
    w_a, w_b = rng.normal(size=3), rng.normal(size=2)
    interaction = rng.normal(size=(3, 2))
    result = migrate_onehot_pair_to_sum(
        ohe_a,
        ohe_b,
        intercept=0.1,
        main_effects_a=w_a,
        main_effects_b=w_b,
        interaction=interaction,
        field_a='fa',
        field_b='fb',
        on_conflict='return',
    )
    assert result.report.max_abs_diff_declared < 1e-9
    df = pd.DataFrame({'fa': ['a', None, 'b'], 'fb': ['x', 'y', 'x']})
    expected = old_scores(df, 'fa', 'fb', ohe_a, ohe_b, 0.1, w_a, w_b, interaction)
    np.testing.assert_allclose(result.score(df), expected, rtol=1e-9, atol=1e-12)

    nan_conflicts = [
        c
        for c in result.conflicts
        if c.kind == 'missing' and pd.isna(c.value_a) and c.value_a is not None and c.value_b == 'x'
    ]
    assert nan_conflicts
    # the new encoder cannot tell nan apart from the declared None level
    none_row_score = result.score(pd.DataFrame({'fa': [None], 'fb': ['x']}))[0]
    assert nan_conflicts[0].new_score == pytest.approx(none_row_score, rel=1e-9)


# ---------------------------------------------------------------------------
# old-encoder variants
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings('ignore:Found unknown categories')
def test_old_encoder_with_drop_first(rng):
    """A dropped first level in the old encoder migrates exactly."""
    ohe_a = make_ohe(['a', 'b', 'c'], drop='first', handle_unknown='ignore')
    ohe_b = make_ohe(['x', 'y'], handle_unknown='ignore')
    w_a, w_b = rng.normal(size=2), rng.normal(size=2)
    interaction = rng.normal(size=(2, 2))
    result = migrate_onehot_pair_to_sum(
        ohe_a,
        ohe_b,
        intercept=0.9,
        main_effects_a=w_a,
        main_effects_b=w_b,
        interaction=interaction,
        field_a='fa',
        field_b='fb',
        on_conflict='return',
    )
    assert result.report.max_abs_diff_declared < 1e-9
    df = pd.DataFrame({'fa': ['a', 'b', 'c'], 'fb': ['x', 'y', 'x']})
    expected = old_scores(df, 'fa', 'fb', ohe_a, ohe_b, 0.9, w_a, w_b, interaction)
    np.testing.assert_allclose(result.score(df), expected, rtol=1e-9, atol=1e-12)


def test_old_encoder_with_infrequent_categories(rng):
    """Infrequent-category lumping of the old encoder migrates exactly."""
    values = ['a'] * 5 + ['b'] * 5 + ['rare1', 'rare2']
    ohe_a = make_ohe(values, min_frequency=2, handle_unknown='infrequent_if_exist')
    ohe_b = make_ohe(['x', 'y'], handle_unknown='ignore')
    w_a, w_b = rng.normal(size=3), rng.normal(size=2)
    interaction = rng.normal(size=(3, 2))
    result = migrate_onehot_pair_to_sum(
        ohe_a,
        ohe_b,
        intercept=0.2,
        main_effects_a=w_a,
        main_effects_b=w_b,
        interaction=interaction,
        field_a='fa',
        field_b='fb',
        on_conflict='return',
    )
    assert result.report.max_abs_diff_declared < 1e-9
    df = pd.DataFrame({'fa': ['a', 'rare1', 'rare2', 'b'], 'fb': ['x', 'y', 'x', 'y']})
    expected = old_scores(df, 'fa', 'fb', ohe_a, ohe_b, 0.2, w_a, w_b, interaction)
    np.testing.assert_allclose(result.score(df), expected, rtol=1e-9, atol=1e-12)


def test_old_encoder_fitted_with_feature_names(rng):
    """Old encoders fitted on DataFrames default the new field names."""
    ohe_a = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(
        pd.DataFrame({'color': ['r', 'g', 'b']})
    )
    ohe_b = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(
        pd.DataFrame({'size': ['s', 'm']})
    )
    w_a, w_b = rng.normal(size=3), rng.normal(size=2)
    interaction = rng.normal(size=(3, 2))
    result = migrate_onehot_pair_to_sum(
        ohe_a,
        ohe_b,
        intercept=1.5,
        main_effects_a=w_a,
        main_effects_b=w_b,
        interaction=interaction,
        on_conflict='return',
    )
    # field names default to the fitted feature names
    assert result.field_a == 'color' and result.field_b == 'size'
    assert result.report.max_abs_diff_declared < 1e-9
    df = pd.DataFrame({'color': ['r', 'b'], 'size': ['s', 'm']})
    expected = old_scores(df, 'color', 'size', ohe_a, ohe_b, 1.5, w_a, w_b, interaction)
    np.testing.assert_allclose(result.score(df), expected, rtol=1e-9, atol=1e-12)


def test_category_encoders_onehot_as_old_encoder(rng):
    """category_encoders' own OneHotEncoder works as the old encoder, too."""
    frame = pd.DataFrame({'fa': ['a', 'b', 'c', 'a'], 'other': [1, 2, 3, 4]})
    old_a = ce.OneHotEncoder(cols=['fa'], use_cat_names=True).fit(frame)
    old_b = ce.OneHotEncoder(cols=['fb']).fit(pd.DataFrame({'fb': ['x', 'y', 'y']}))
    cols_a = [c for c in old_a.get_feature_names_out() if c.startswith('fa_')]
    cols_b = [c for c in old_b.get_feature_names_out() if c.startswith('fb_')]
    w_a, w_b = rng.normal(size=len(cols_a)), rng.normal(size=len(cols_b))
    interaction = rng.normal(size=(len(cols_a), len(cols_b)))
    result = migrate_onehot_pair_to_sum(
        old_a,
        old_b,
        intercept=0.6,
        main_effects_a=w_a,
        main_effects_b=w_b,
        interaction=interaction,
        field_a='fa',
        field_b='fb',
        on_conflict='return',
    )
    assert result.report.max_abs_diff_declared < 1e-9

    def ce_scores(df):
        enc_a = old_a.transform(df.assign(other=0))
        enc_b = old_b.transform(df)
        z_a = enc_a[cols_a].to_numpy(dtype=float)
        z_b = enc_b[cols_b].to_numpy(dtype=float)
        return 0.6 + z_a @ w_a + z_b @ w_b + np.einsum('ij,jk,ik->i', z_a, interaction, z_b)

    df = pd.DataFrame({'fa': ['a', 'c', 'b'], 'fb': ['x', 'y', 'x']})
    np.testing.assert_allclose(result.score(df), ce_scores(df), rtol=1e-9, atol=1e-12)


def test_numeric_categories_with_declared_nan(rng):
    """Numeric categories, including a declared nan, migrate exactly."""
    ohe_a = make_ohe([1.0, 2.0, np.nan], handle_unknown='ignore')
    ohe_b = make_ohe([10.0, 20.0], handle_unknown='ignore')
    w_a, w_b = rng.normal(size=3), rng.normal(size=2)
    interaction = rng.normal(size=(3, 2))
    result = migrate_onehot_pair_to_sum(
        ohe_a,
        ohe_b,
        intercept=-0.3,
        main_effects_a=w_a,
        main_effects_b=w_b,
        interaction=interaction,
        field_a='fa',
        field_b='fb',
        on_conflict='return',
    )
    assert result.report.max_abs_diff_declared < 1e-9
    df = pd.DataFrame({'fa': [1.0, np.nan, 2.0], 'fb': [10.0, 20.0, 10.0]})
    expected = old_scores(df, 'fa', 'fb', ohe_a, ohe_b, -0.3, w_a, w_b, interaction)
    np.testing.assert_allclose(result.score(df), expected, rtol=1e-9, atol=1e-12)


def test_single_category_field(rng):
    """A field with one declared category gets zero encoded columns."""
    ohe_a = make_ohe(['only'], handle_unknown='ignore')
    ohe_b = make_ohe(['x', 'y'], handle_unknown='ignore')
    w_a, w_b = rng.normal(size=1), rng.normal(size=2)
    result = migrate_onehot_pair_to_sum(
        ohe_a,
        ohe_b,
        intercept=0.5,
        main_effects_a=w_a,
        main_effects_b=w_b,
        interaction=rng.normal(size=(1, 2)),
        field_a='fa',
        field_b='fb',
        on_conflict='return',
    )
    assert result.feature_names_a == []
    assert result.main_effects_a.shape == (0,)
    assert result.interaction.shape == (0, 1)
    assert result.report.max_abs_diff_declared < 1e-9
    df = pd.DataFrame({'fa': ['only', 'only'], 'fb': ['x', 'y']})
    assert result.score(df).shape == (2,)


# ---------------------------------------------------------------------------
# determinism, column order, defaults
# ---------------------------------------------------------------------------


def test_migration_is_deterministic(rng):
    """Two identical migrations produce identical coefficients."""
    ohe_a = make_ohe(['a', 'b', 'c'], handle_unknown='ignore')
    ohe_b = make_ohe(['x', 'y'], handle_unknown='ignore')
    w_a, w_b = rng.normal(size=3), rng.normal(size=2)
    interaction = rng.normal(size=(3, 2))
    kwargs = {
        'intercept': 0.7,
        'main_effects_a': w_a,
        'main_effects_b': w_b,
        'interaction': interaction,
        'field_a': 'fa',
        'field_b': 'fb',
        'on_conflict': 'return',
    }
    first = migrate_onehot_pair_to_sum(ohe_a, ohe_b, **kwargs)
    second = migrate_onehot_pair_to_sum(ohe_a, ohe_b, **kwargs)
    assert first.intercept == second.intercept
    for left, right in [
        (first.main_effects_a, second.main_effects_a),
        (first.main_effects_b, second.main_effects_b),
        (first.interaction, second.interaction),
    ]:
        np.testing.assert_array_equal(left, right)


def test_flat_interaction_matches_matrix(rng):
    """A flat interaction vector is read in C order and matches the matrix form."""
    ohe_a = make_ohe(['a', 'b', 'c'], handle_unknown='ignore')
    ohe_b = make_ohe(['x', 'y'], handle_unknown='ignore')
    w_a, w_b = rng.normal(size=3), rng.normal(size=2)
    interaction = rng.normal(size=(3, 2))
    kwargs = {
        'intercept': 0.7,
        'main_effects_a': w_a,
        'main_effects_b': w_b,
        'field_a': 'fa',
        'field_b': 'fb',
        'on_conflict': 'return',
    }
    as_matrix = migrate_onehot_pair_to_sum(ohe_a, ohe_b, interaction=interaction, **kwargs)
    as_flat = migrate_onehot_pair_to_sum(ohe_a, ohe_b, interaction=interaction.ravel(), **kwargs)
    np.testing.assert_array_equal(as_matrix.interaction, as_flat.interaction)
    assert as_matrix.intercept == as_flat.intercept


def test_missing_coefficients_default_to_zero():
    """Omitted main effects and interactions are treated as zero."""
    ohe_a = make_ohe(['a', 'b'], handle_unknown='ignore')
    ohe_b = make_ohe(['x', 'y'], handle_unknown='ignore')
    result = migrate_onehot_pair_to_sum(
        ohe_a, ohe_b, intercept=2.0, field_a='fa', field_b='fb', on_conflict='return'
    )
    assert result.intercept == pytest.approx(2.0)
    np.testing.assert_array_equal(result.interaction, np.zeros((1, 1)))
    df = pd.DataFrame({'fa': ['a', 'b'], 'fb': ['x', 'y']})
    np.testing.assert_allclose(result.score(df), [2.0, 2.0])


def test_native_column_order_is_preserved(rng):
    """Coefficients align with the encoders' native output column order."""
    ohe_a = make_ohe(['a', 'b', 'c'], handle_unknown='ignore')
    ohe_b = make_ohe(['x', 'y'], handle_unknown='ignore')
    result = migrate_onehot_pair_to_sum(
        ohe_a,
        ohe_b,
        intercept=0.7,
        main_effects_a=rng.normal(size=3),
        main_effects_b=rng.normal(size=2),
        interaction=rng.normal(size=(3, 2)),
        field_a='fa',
        field_b='fb',
        on_conflict='return',
    )
    df = pd.DataFrame({'fa': ['b', 'a'], 'fb': ['y', 'x'], 'extra': [1, 2]})
    encoded = result.encoder_a.transform(df)
    generated = [c for c in encoded.columns if c.startswith('fa_')]
    assert result.feature_names_a == generated
    # scoring works directly on raw rows with extra pass-through columns
    z_a = encoded[result.feature_names_a].to_numpy()
    z_b = result.encoder_b.transform(df)[result.feature_names_b].to_numpy()
    manual = (
        result.intercept
        + z_a @ result.main_effects_a
        + z_b @ result.main_effects_b
        + np.einsum('ij,jk,ik->i', z_a, result.interaction, z_b)
    )
    np.testing.assert_allclose(result.score(df), manual, rtol=1e-12, atol=1e-12)


def test_new_encoders_are_fit_on_declared_categories_only():
    """The new encoders know exactly the declared categories, no sample data."""
    ohe_a = make_ohe(['b', 'a', 'c'], handle_unknown='ignore')
    ohe_b = make_ohe(['x', 'y'], handle_unknown='ignore')
    result = migrate_onehot_pair_to_sum(
        ohe_a, ohe_b, intercept=0.0, field_a='fa', field_b='fb', on_conflict='return'
    )
    (mapping,) = result.encoder_a.ordinal_encoder.category_mapping
    declared = set(mapping['mapping'][mapping['mapping'] > 0].index)
    assert declared == {'a', 'b', 'c'}
    # the new encoder reproduces the canonical SumEncoder behaviour
    reference = ce.SumEncoder(cols=['fa']).fit(pd.DataFrame({'fa': ['a', 'b', 'c']}))
    probe = pd.DataFrame({'fa': ['a', 'b', 'c', 'unseen', np.nan]})
    pd.testing.assert_frame_equal(result.encoder_a.transform(probe), reference.transform(probe))


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_validation_of_coefficient_shapes():
    """Coefficient blocks must match the old encoders' output widths."""
    ohe_a = make_ohe(['a', 'b', 'c'])
    ohe_b = make_ohe(['x', 'y'])
    with pytest.raises(ValueError, match='main_effects_a'):
        migrate_onehot_pair_to_sum(
            ohe_a, ohe_b, intercept=0.0, main_effects_a=[1.0, 2.0], field_b='fb'
        )
    with pytest.raises(ValueError, match='interaction'):
        migrate_onehot_pair_to_sum(
            ohe_a, ohe_b, intercept=0.0, interaction=np.zeros((2, 2)), field_b='fb'
        )
    with pytest.raises(ValueError, match='finite'):
        migrate_onehot_pair_to_sum(ohe_a, ohe_b, intercept=np.nan, field_b='fb')


def test_validation_of_encoders():
    """Unfitted, multi-column or foreign encoders are rejected clearly."""
    ohe_b = make_ohe(['x', 'y'])
    with pytest.raises(ValueError, match='not fitted'):
        migrate_onehot_pair_to_sum(
            OneHotEncoder(), ohe_b, intercept=0.0, field_a='fa', field_b='fb'
        )
    multi = OneHotEncoder().fit(np.array([['a', 'x'], ['b', 'y']], dtype=object))
    with pytest.raises(ValueError, match='exactly one column'):
        migrate_onehot_pair_to_sum(multi, ohe_b, intercept=0.0, field_a='fa', field_b='fb')
    with pytest.raises(ValueError, match='OneHotEncoder'):
        migrate_onehot_pair_to_sum('nope', ohe_b, intercept=0.0, field_a='fa', field_b='fb')


def test_validation_of_field_names_and_on_conflict():
    """Field names must differ and on_conflict must be a known mode."""
    ohe_a = make_ohe(['a', 'b'])
    ohe_b = make_ohe(['x', 'y'])
    with pytest.raises(ValueError, match='different column names'):
        migrate_onehot_pair_to_sum(ohe_a, ohe_b, intercept=0.0, field_a='same', field_b='same')
    with pytest.raises(ValueError, match='on_conflict'):
        migrate_onehot_pair_to_sum(
            ohe_a, ohe_b, intercept=0.0, field_a='fa', field_b='fb', on_conflict='ignore'
        )


def test_score_input_validation(rng):
    """score() requires a DataFrame carrying both migrated fields."""
    ohe_a = make_ohe(['a', 'b'])
    ohe_b = make_ohe(['x', 'y'])
    result = migrate_onehot_pair_to_sum(
        ohe_a, ohe_b, intercept=0.0, field_a='fa', field_b='fb', on_conflict='return'
    )
    with pytest.raises(TypeError, match='DataFrame'):
        result.score(np.array([['a', 'x']], dtype=object))
    with pytest.raises(ValueError, match='missing'):
        result.score(pd.DataFrame({'fa': ['a']}))


# ---------------------------------------------------------------------------
# value equality semantics (None vs nan are distinct for sklearn)
# ---------------------------------------------------------------------------


def test_values_equal_distinguishes_none_from_nan():
    """None and nan are distinct categories; nan-like values compare equal."""
    assert _values_equal(np.nan, float('nan'))
    assert _values_equal(None, None)
    assert not _values_equal(None, np.nan)
    assert not _values_equal(np.nan, 'x')
    assert _values_equal('a', 'a')
    assert _values_equal(1, 1.0)
    assert not _values_equal('a', 'b')
