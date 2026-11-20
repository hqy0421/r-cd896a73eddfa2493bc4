"""Re-express a deployed one-hot linear score in :class:`SumEncoder` space.

This module provides :func:`migrate_onehot_pair_to_sum`, a migration tool for
deployed linear scoring pipelines whose training data is no longer available.
It reads two fitted one-hot encoders (one per categorical field), the old
score's intercept, main effects and the two-field interaction block, and
produces the intercept, main effects and interaction of an equivalent linear
score on top of :class:`~category_encoders.sum_coding.SumEncoder` features.

No retraining is involved: the new encoders are fitted on the declared
category lists read from the old encoders (a structural fit that only fixes
the contrast matrices), and the new coefficients are computed algebraically
from the old coefficients. The migration is verified by actually transforming
every combination of declared categories - including declared missing values
and unseen probe inputs - with both the old and the new encoders and comparing
the old and new linear scores. Combinations that cannot be represented
exactly are reported as explicit conflicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from category_encoders.sum_coding import SumEncoder

__all__ = [
    'MigrationConflict',
    'SumMigrationConflictError',
    'SumMigrationReport',
    'SumMigrationResult',
    'migrate_onehot_pair_to_sum',
]

_DECLARED = 'declared'
_MISSING = 'missing'
_UNSEEN = 'unseen'

# Cap on stored conflicts so pathological grids cannot exhaust memory; the
# report notes when truncation happened.
_MAX_STORED_CONFLICTS = 1000


def _is_missing(value: Any) -> bool:
    """True for nan-like scalars (np.nan, None, pd.NA, pd.NaT)."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _values_equal(a: Any, b: Any) -> bool:
    """Equality that treats nan-like values as equal to each other, None only to None."""
    if a is b:
        return True
    a_missing, b_missing = _is_missing(a), _is_missing(b)
    if a_missing != b_missing:
        return False
    if a_missing:
        # sklearn keeps None and nan as distinct categories in object arrays,
        # so only equate None with None and nan-like with nan-like.
        return (a is None) == (b is None)
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def _contains(values: Sequence, value: Any) -> bool:
    return any(_values_equal(v, value) for v in values)


class _OldOneHotView:
    """Uniform read-only access to a fitted single-column one-hot encoder.

    Supports scikit-learn's ``OneHotEncoder`` (duck-typed via ``categories_``)
    and this package's own :class:`~category_encoders.one_hot.OneHotEncoder`.
    """

    def __init__(self, encoder: Any, role: str):
        self.encoder = encoder
        is_sklearn = hasattr(encoder, 'categories_') or type(encoder).__module__.startswith(
            'sklearn.'
        )
        if is_sklearn:
            self._init_sklearn(encoder, role)
        elif hasattr(encoder, 'ordinal_encoder') and hasattr(encoder, 'mapping'):
            self._init_category_encoders(encoder, role)
        else:
            raise ValueError(
                f'{role}: expected a fitted single-column OneHotEncoder '
                '(sklearn.preprocessing.OneHotEncoder or '
                'category_encoders.OneHotEncoder), got '
                f'{type(encoder).__name__!r}.'
            )
        # authoritative output width, from an actual transform
        self.n_out = int(self.transform_values(self.categories[:1]).shape[1])

    def _init_sklearn(self, encoder: Any, role: str) -> None:
        categories = getattr(encoder, 'categories_', None)
        if categories is None:
            raise ValueError(f'{role}: the OneHotEncoder is not fitted (no categories_).')
        if len(categories) != 1:
            raise ValueError(
                f'{role}: expected a OneHotEncoder fitted on exactly one column, '
                f'got {len(categories)} columns. Fit one encoder per field and pass '
                'them separately.'
            )
        self.kind = 'sklearn'
        self.categories = list(categories[0])
        names_in = getattr(encoder, 'feature_names_in_', None)
        self.feature_name = str(names_in[0]) if names_in is not None else None
        self.desc = (
            f'sklearn OneHotEncoder(handle_unknown={getattr(encoder, "handle_unknown", "?")!r})'
        )

    def _init_category_encoders(self, encoder: Any, role: str) -> None:
        if getattr(encoder, 'mapping', None) is None:
            raise ValueError(f'{role}: the OneHotEncoder is not fitted (mapping is None).')
        cols = list(getattr(encoder, 'cols', None) or [])
        if len(cols) != 1:
            raise ValueError(
                f'{role}: expected a OneHotEncoder fitted on exactly one column, '
                f'got cols={cols}. Fit one encoder per field and pass them separately.'
            )
        self.kind = 'category_encoders'
        self.column = cols[0]
        self.feature_name = self.column
        self.desc = 'category_encoders OneHotEncoder'
        # fitted input columns, to pad probe frames to the fitted width (the
        # other columns are pass-through: cols is required to be single-column)
        self._fit_columns = list(getattr(encoder, 'feature_names_in_', None) or [self.column])
        encoded = None
        for switch in encoder.mapping:
            if switch.get('col') == self.column:
                encoded = switch.get('mapping')
                break
        if encoded is None:
            raise ValueError(f'{role}: no fitted mapping found for column {self.column!r}.')
        self.encoded_columns = list(encoded.columns)
        ordinal_mapping = None
        for switch in encoder.ordinal_encoder.category_mapping:
            if switch.get('col') == self.column:
                ordinal_mapping = switch.get('mapping')
                break
        if ordinal_mapping is None:
            raise ValueError(f'{role}: no fitted ordinal mapping found for column {self.column!r}.')
        declared = ordinal_mapping[ordinal_mapping > 0].sort_values()
        self.categories = list(declared.index)

    def transform_values(self, values: Sequence) -> np.ndarray:
        """Encode a list of raw values into a dense (n_values, n_out) matrix."""
        values = list(values)
        if self.kind == 'sklearn':
            if self.feature_name is not None:
                X = pd.DataFrame({self.feature_name: pd.Series(values, dtype=object)})
            else:
                X = np.empty((len(values), 1), dtype=object)
                X[:, 0] = values
            encoded = self.encoder.transform(X)
            if sparse.issparse(encoded):
                encoded = encoded.toarray()
            return np.asarray(encoded, dtype=float)
        data = {name: pd.Series([0] * len(values), dtype=object) for name in self._fit_columns}
        data[self.column] = pd.Series(values, dtype=object)
        X = pd.DataFrame(data)[self._fit_columns]
        out = self.encoder.transform(X)
        return out[self.encoded_columns].to_numpy(dtype=float)


def _fit_sum_encoder(field: str, categories: Sequence) -> SumEncoder:
    """Fit a SumEncoder on the declared categories of one field.

    This is a structural fit: it only fixes the category-to-contrast-row
    assignment from the categories declared by the old encoder. No training
    data is used.
    """
    encoder = SumEncoder(cols=[field])
    encoder.fit(pd.DataFrame({field: pd.Series(list(categories))}))
    return encoder


def _encoded_columns(encoder: SumEncoder, field: str) -> list[str]:
    """Encoded column names of one field, in the encoder's native output order."""
    for switch in encoder.mapping or []:
        if switch.get('col') == field:
            return list(switch['mapping'].columns)
    return []


def _sum_matrix(
    encoder: SumEncoder, field: str, columns: list[str], values: Sequence
) -> np.ndarray:
    """Encode values with the new encoder, returning a dense (n_values, k) matrix."""
    values = list(values)
    if not columns:
        return np.zeros((len(values), 0))
    out = encoder.transform(pd.DataFrame({field: pd.Series(values)}))
    return out[columns].to_numpy(dtype=float)


def _as_effect_vector(coef: Any, width: int, name: str) -> np.ndarray:
    if coef is None:
        return np.zeros(width)
    vec = np.asarray(coef, dtype=float).ravel()
    if vec.shape[0] != width:
        raise ValueError(
            f'{name}: expected {width} coefficients (the output width of the old '
            f'encoder), got {vec.shape[0]}.'
        )
    if not np.all(np.isfinite(vec)):
        raise ValueError(f'{name}: coefficients must be finite.')
    return vec


def _as_interaction_matrix(interaction: Any, rows: int, cols: int) -> np.ndarray:
    if interaction is None:
        return np.zeros((rows, cols))
    mat = np.asarray(interaction, dtype=float)
    if mat.ndim == 1:
        if mat.size != rows * cols:
            raise ValueError(
                f'interaction: expected a ({rows}, {cols}) matrix or a flat vector of '
                f'size {rows * cols}, got a flat vector of size {mat.size}.'
            )
        mat = mat.reshape(rows, cols)
    if mat.shape != (rows, cols):
        raise ValueError(
            f'interaction: expected shape ({rows}, {cols}) aligned with the old '
            f'encoders of field A and B, got {mat.shape}.'
        )
    if not np.all(np.isfinite(mat)):
        raise ValueError('interaction: coefficients must be finite.')
    return mat


def _solve_effect(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve ``matrix @ coef = target`` exactly (deterministic least squares)."""
    if matrix.shape[1] == 0:
        return np.zeros(0)
    coef, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    return coef


def _solve_interaction(mat_a: np.ndarray, mat_b: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve ``mat_a @ coef @ mat_b.T = target`` exactly (deterministic)."""
    k_a, k_b = mat_a.shape[1], mat_b.shape[1]
    if k_a == 0 or k_b == 0:
        return np.zeros((k_a, k_b))
    step, *_ = np.linalg.lstsq(mat_a, target, rcond=None)
    coef_t, *_ = np.linalg.lstsq(mat_b, step.T, rcond=None)
    return coef_t.T


def _auto_unseen_value(categories: list) -> Any:
    """One probe value that is guaranteed not to be a declared category."""
    non_missing = [c for c in categories if not _is_missing(c)]
    numeric = non_missing and all(
        isinstance(c, (int, float, np.integer, np.floating)) and not isinstance(c, (bool, np.bool_))
        for c in non_missing
    )
    if numeric:
        candidate = max(float(c) for c in non_missing) + 1.0
        return candidate if np.isfinite(candidate) else None
    candidate = '__UNSEEN_CATEGORY__'
    while _contains(categories, candidate):
        candidate += '_'
    return candidate


def _probe_values(categories: list, user_unseen: Sequence) -> list:
    """Missing and unseen probe values extending the declared grid."""
    probes = []
    # sklearn keeps None and nan as distinct categories in object arrays, so a
    # declared None does not cover undeclared nan and vice versa.
    has_nan_like = any(_is_missing(c) and c is not None for c in categories)
    if not has_nan_like:
        probes.append(np.nan)
    non_missing = [c for c in categories if not _is_missing(c)]
    all_numeric = non_missing and all(
        isinstance(c, (int, float, np.integer, np.floating)) and not isinstance(c, (bool, np.bool_))
        for c in non_missing
    )
    # None is a distinct category from nan for sklearn object arrays, while the
    # SumEncoder merges it with nan: worth probing whenever it is not declared.
    if not all_numeric and not any(c is None for c in categories):
        probes.append(None)
    for value in user_unseen:
        if not _contains(categories, value) and not _contains(probes, value):
            probes.append(value)
    auto = _auto_unseen_value(categories)
    if auto is not None and not _contains(probes, auto):
        probes.append(auto)
    return probes


def _row_desc(row: np.ndarray) -> str:
    if row.size == 0:
        return 'an empty row (no encoded columns)'
    if np.all(row == 0):
        return 'an all-zero row'
    return f'row {np.round(row, 6).tolist()}'


def _cell_kind(kind_a: str, kind_b: str) -> str:
    if kind_a == _DECLARED and kind_b == _DECLARED:
        return _DECLARED
    if _MISSING in (kind_a, kind_b):
        return _MISSING
    return _UNSEEN


@dataclass
class MigrationConflict:
    """One input combination whose old and new linear scores differ.

    Attributes
    ----------
    field_a, field_b: str
        Names of the two migrated fields.
    value_a, value_b:
        The raw input values of the conflicting combination (None for
        structural conflicts that are not tied to one combination).
    kind: str
        'declared', 'missing', 'unseen' or 'unrepresentable'.
    old_score, new_score: float or None
        The two linear scores for this combination (None when not applicable).
    abs_diff: float
        Absolute score difference (residual for structural conflicts).
    reason: str
        Human-readable explanation of why the scores differ.
    """

    field_a: str
    field_b: str
    value_a: Any
    value_b: Any
    kind: str
    old_score: float | None
    new_score: float | None
    abs_diff: float
    reason: str


@dataclass
class SumMigrationReport:
    """Verification outcome of a migration, produced by actual transforms.

    Attributes
    ----------
    n_declared_cells: int
        Number of declared-category combinations (the full declared grid).
    n_checked_cells: int
        Number of grid cells compared (declared grid plus missing/unseen
        probes, minus cells the old encoder cannot score).
    n_unscorable_cells: int
        Number of probe cells the old encoder rejected (e.g. with
        ``handle_unknown='error'``); these are listed in ``notes``.
    max_abs_diff_declared: float
        Largest absolute old-vs-new score difference on the declared grid.
    max_abs_diff: float
        Largest absolute difference over all checked cells.
    conflicts: list of MigrationConflict
        Every checked combination whose scores differ beyond tolerance
        (capped at 1000 entries; truncation is noted in ``notes``).
    notes: list of str
        Informational messages, e.g. probe values the old encoder rejects.
    """

    n_declared_cells: int
    n_checked_cells: int
    n_unscorable_cells: int
    max_abs_diff_declared: float
    max_abs_diff: float
    conflicts: list[MigrationConflict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class SumMigrationResult:
    """Migrated coefficients and the encoders of the new scoring space.

    The coefficient arrays are laid out in the native output column order of
    the fitted :class:`~category_encoders.sum_coding.SumEncoder` objects, so
    ``encoder_a.transform(X)[feature_names_a] @ main_effects_a`` is a valid
    scoring expression. :meth:`score` does this for raw input rows directly.

    Attributes
    ----------
    encoder_a, encoder_b: SumEncoder
        The new encoders, fitted on the declared categories of the old ones.
    field_a, field_b: str
        Column names the new encoders and :meth:`score` operate on.
    intercept: float
        New intercept (grand mean of the old score over the declared grid).
    main_effects_a, main_effects_b: np.ndarray
        New main-effect coefficients, aligned with ``feature_names_a`` /
        ``feature_names_b``.
    interaction: np.ndarray
        New interaction matrix of shape
        ``(len(feature_names_a), len(feature_names_b))``;
        ``interaction[i, j]`` multiplies the i-th encoded column of field A
        with the j-th encoded column of field B.
    feature_names_a, feature_names_b: list of str
        Encoded column names in the encoders' native output order.
    declared_categories_a, declared_categories_b: list
        The declared categories the migration is exact for.
    report: SumMigrationReport
        The verification report.
    """

    encoder_a: SumEncoder
    encoder_b: SumEncoder
    field_a: str
    field_b: str
    intercept: float
    main_effects_a: np.ndarray
    main_effects_b: np.ndarray
    interaction: np.ndarray
    feature_names_a: list[str]
    feature_names_b: list[str]
    declared_categories_a: list
    declared_categories_b: list
    report: SumMigrationReport

    @property
    def ok(self) -> bool:
        """True when no conflict was found on any checked input combination."""
        return not self.report.conflicts

    @property
    def conflicts(self) -> list[MigrationConflict]:
        """The conflicts of the verification report (empty when exact)."""
        return self.report.conflicts

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Score raw rows with the migrated coefficients.

        Parameters
        ----------
        X: pd.DataFrame
            Raw input rows containing the columns ``field_a`` and ``field_b``
            (extra columns are passed through the encoders untouched).

        Returns
        -------
        np.ndarray
            The linear score of each row: ``intercept`` plus main effects plus
            the two-field interaction, in the new SumEncoder space.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f'score() expects a pandas DataFrame with columns {self.field_a!r} '
                f'and {self.field_b!r}, got {type(X).__name__!r}.'
            )
        missing_cols = [c for c in (self.field_a, self.field_b) if c not in X.columns]
        if missing_cols:
            raise ValueError(f'X is missing the required columns: {missing_cols}.')
        z_a = self.encoder_a.transform(X)
        z_b = self.encoder_b.transform(X)
        scores = np.full(len(X), self.intercept, dtype=float)
        mat_a = (
            z_a[self.feature_names_a].to_numpy(dtype=float)
            if self.feature_names_a
            else np.zeros((len(X), 0))
        )
        mat_b = (
            z_b[self.feature_names_b].to_numpy(dtype=float)
            if self.feature_names_b
            else np.zeros((len(X), 0))
        )
        scores += mat_a @ self.main_effects_a + mat_b @ self.main_effects_b
        if self.interaction.size:
            scores += np.einsum('ij,jk,ik->i', mat_a, self.interaction, mat_b)
        return scores


class SumMigrationConflictError(ValueError):
    """Raised when no exactly equivalent migration to SumEncoder space exists.

    Carries the structured conflicts in ``conflicts``; call
    :func:`migrate_onehot_pair_to_sum` with ``on_conflict='return'`` to get the
    migrated coefficients together with the conflict report instead.
    """

    def __init__(self, message: str, conflicts: list[MigrationConflict]):
        super().__init__(message)
        self.conflicts = list(conflicts)


def _transform_extended(
    view: _OldOneHotView, values: list, n_declared: int
) -> tuple[np.ndarray, np.ndarray, dict[int, Exception]]:
    """Transform declared values in one batch and probes one by one.

    Per-value probing captures the old encoder's exact behaviour (including
    raised errors) for missing and unseen inputs.
    """
    rows = np.zeros((len(values), view.n_out))
    ok = np.ones(len(values), dtype=bool)
    errors: dict[int, Exception] = {}
    try:
        rows[:n_declared] = view.transform_values(values[:n_declared])
    except Exception:
        # fall back to per-value transforms so one bad value cannot hide others
        for i in range(n_declared):
            try:
                rows[i] = view.transform_values([values[i]])[0]
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                ok[i] = False
                errors[i] = exc
    for i in range(n_declared, len(values)):
        try:
            rows[i] = view.transform_values([values[i]])[0]
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            ok[i] = False
            errors[i] = exc
    return rows, ok, errors


def migrate_onehot_pair_to_sum(
    old_encoder_a: Any,
    old_encoder_b: Any,
    *,
    intercept: float,
    main_effects_a: Any = None,
    main_effects_b: Any = None,
    interaction: Any = None,
    field_a: str | None = None,
    field_b: str | None = None,
    unseen_values_a: Sequence = (),
    unseen_values_b: Sequence = (),
    rtol: float = 1e-7,
    atol: float = 1e-9,
    on_conflict: Literal['raise', 'return'] = 'raise',
) -> SumMigrationResult:
    """Migrate a deployed two-field linear score from one-hot to SumEncoder space.

    Reads two fitted one-hot encoders (sklearn's or category_encoders'
    ``OneHotEncoder``, one column each), the old score's intercept, the main
    effects of both fields and their interaction block, and produces the
    intercept, main effects and interaction of an equivalent linear score on
    :class:`~category_encoders.sum_coding.SumEncoder` features. No training
    data is used and the old model is never refitted: the new encoders are
    fitted on the declared categories of the old encoders (a structural fit
    that only fixes the contrast matrices) and the new coefficients are
    computed algebraically.

    On the grid of declared category combinations the new score is an exact
    re-parametrization of the old one: sum-contrast columns plus an intercept
    span the same function space as one-hot columns, so the solution exists
    and is unique (for degenerate inputs the deterministic minimum-norm
    solution is used, so repeated runs are identical). For missing and unseen
    inputs exact equivalence depends on the old encoder's semantics: the new
    SumEncoder encodes both as all-zero rows, i.e. as the average of the
    declared-level scores. Every combination - declared, declared-missing and
    unseen probes - is checked by actually transforming it with both the old
    and the new encoders and comparing the old and new linear scores;
    combinations that cannot be matched are reported as conflicts.

    Parameters
    ----------
    old_encoder_a, old_encoder_b:
        The two fitted one-hot encoders of the deployed score, one per field.
    intercept: float
        Intercept of the old linear score. It is absorbed into the migrated
        coefficients: the returned ``intercept`` replaces it (coefficients of
        any other, non-migrated model terms stay unchanged).
    main_effects_a, main_effects_b: array-like, optional
        The old main-effect coefficients, aligned with the output columns of
        the old encoders (length = each encoder's output width). Defaults to
        zero when omitted.
    interaction: array-like, optional
        The old interaction block as a ``(n_out_a, n_out_b)`` matrix (or a
        flat vector of that size, read in C order): ``interaction[i, j]``
        multiplies the i-th encoded column of field A with the j-th encoded
        column of field B. Defaults to zero when omitted.
    field_a, field_b: str, optional
        Column names for the new encoders and for :meth:`SumMigrationResult.score`.
        Defaults to the old encoders' fitted feature names, else 'field_a' /
        'field_b'.
    unseen_values_a, unseen_values_b: sequence, optional
        Extra unseen probe values to verify, in addition to one automatically
        generated unseen value per field.
    rtol, atol: float
        Tolerances of the old-vs-new score comparison, combined as
        ``|diff| <= atol + rtol * |old_score|``.
    on_conflict: 'raise' or 'return'
        'raise' (default) raises :class:`SumMigrationConflictError` when exact
        equivalence is impossible; 'return' returns the result with the
        conflicts attached to ``result.report``.

    Returns
    -------
    SumMigrationResult
        The migrated coefficients, the fitted SumEncoders and the
        verification report. ``result.score(df)`` scores raw new rows
        directly, with coefficients in the encoders' native column order.

    Raises
    ------
    SumMigrationConflictError
        When ``on_conflict='raise'`` and exact equivalence is impossible.
        The exception carries the structured ``conflicts``.
    ValueError
        For invalid inputs (unfitted or multi-column encoders, coefficient
        blocks that do not match the old encoders' output widths, ...).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.preprocessing import OneHotEncoder
    >>> from category_encoders.migration import migrate_onehot_pair_to_sum
    >>> ohe_a = OneHotEncoder(sparse_output=False).fit(pd.DataFrame({'color': ['r', 'g', 'b']}))
    >>> ohe_b = OneHotEncoder(sparse_output=False).fit(pd.DataFrame({'size': ['s', 'm']}))
    >>> result = migrate_onehot_pair_to_sum(
    ...     ohe_a,
    ...     ohe_b,
    ...     intercept=1.5,
    ...     main_effects_a=[0.3, -0.1, -0.2],
    ...     main_effects_b=[0.5, -0.5],
    ...     interaction=[[0.1, -0.1], [0.0, 0.0], [-0.1, 0.1]],
    ...     field_a='color',
    ...     field_b='size',
    ...     on_conflict='return',
    ... )
    >>> result.ok
    True
    >>> result.score(pd.DataFrame({'color': ['r', 'b'], 'size': ['s', 'm']}))
    array([0.9, 2.4])
    """
    if on_conflict not in ('raise', 'return'):
        raise ValueError(f"on_conflict must be 'raise' or 'return', got {on_conflict!r}.")
    view_a = _OldOneHotView(old_encoder_a, 'old_encoder_a')
    view_b = _OldOneHotView(old_encoder_b, 'old_encoder_b')

    w_a = _as_effect_vector(main_effects_a, view_a.n_out, 'main_effects_a')
    w_b = _as_effect_vector(main_effects_b, view_b.n_out, 'main_effects_b')
    mat_interaction = _as_interaction_matrix(interaction, view_a.n_out, view_b.n_out)
    old_intercept = float(intercept)
    if not np.isfinite(old_intercept):
        raise ValueError(f'intercept: expected a finite number, got {intercept!r}.')

    field_a = field_a or view_a.feature_name or 'field_a'
    field_b = field_b or view_b.feature_name or 'field_b'
    if field_a == field_b:
        raise ValueError(
            f'field_a and field_b must be different column names, got {field_a!r} for both.'
        )

    declared_a = view_a.categories
    declared_b = view_b.categories
    if not declared_a or not declared_b:
        raise ValueError('The old encoders declare no categories; nothing to migrate.')

    # --- new encoders: structural fit on the declared categories only -------
    encoder_a = _fit_sum_encoder(field_a, declared_a)
    encoder_b = _fit_sum_encoder(field_b, declared_b)
    names_a = _encoded_columns(encoder_a, field_a)
    names_b = _encoded_columns(encoder_b, field_b)

    # --- actual transforms of every declared category, old and new ----------
    mat_a = _sum_matrix(encoder_a, field_a, names_a, declared_a)
    mat_b = _sum_matrix(encoder_b, field_b, names_b, declared_b)
    oh_a = view_a.transform_values(declared_a)
    oh_b = view_b.transform_values(declared_b)

    # --- old score on the declared grid, then the unique ANOVA decomposition -
    eff_a = oh_a @ w_a
    eff_b = oh_b @ w_b
    grid = old_intercept + eff_a[:, None] + eff_b[None, :] + oh_a @ mat_interaction @ oh_b.T
    new_intercept = float(grid.mean())
    dev_a = grid.mean(axis=1) - new_intercept
    dev_b = grid.mean(axis=0) - new_intercept
    resid = grid - new_intercept - dev_a[:, None] - dev_b[None, :]

    # --- exact coefficients of the new space (unique / deterministic) -------
    new_w_a = _solve_effect(mat_a, dev_a)
    new_w_b = _solve_effect(mat_b, dev_b)
    new_interaction = _solve_interaction(mat_a, mat_b, resid)

    # --- representability check on the declared grid -------------------------
    conflicts: list[MigrationConflict] = []
    scale = max(1.0, float(np.abs(grid).max()))
    rep_tol = 1e-8 * scale
    residuals = [
        ('main effect of field ' + repr(field_a), dev_a - mat_a @ new_w_a),
        ('main effect of field ' + repr(field_b), dev_b - mat_b @ new_w_b),
        ('interaction', resid - mat_a @ new_interaction @ mat_b.T),
    ]
    for what, remainder in residuals:
        remainder = np.asarray(remainder, dtype=float)
        max_residual = float(np.abs(remainder).max()) if remainder.size else 0.0
        if max_residual > rep_tol:
            conflicts.append(
                MigrationConflict(
                    field_a=field_a,
                    field_b=field_b,
                    value_a=None,
                    value_b=None,
                    kind='unrepresentable',
                    old_score=None,
                    new_score=None,
                    abs_diff=max_residual,
                    reason=(
                        f'the {what} of the old score cannot be represented exactly '
                        f'in the SumEncoder space (residual {max_residual:.6g} exceeds '
                        f'{rep_tol:.3g}).'
                    ),
                )
            )

    # --- verification by actual transforms on the extended grid -------------
    probes_a = _probe_values(declared_a, unseen_values_a)
    probes_b = _probe_values(declared_b, unseen_values_b)
    ext_a = declared_a + probes_a
    ext_b = declared_b + probes_b
    kinds_a = [_DECLARED] * len(declared_a) + [
        _MISSING if _is_missing(v) else _UNSEEN for v in probes_a
    ]
    kinds_b = [_DECLARED] * len(declared_b) + [
        _MISSING if _is_missing(v) else _UNSEEN for v in probes_b
    ]

    oh_a_ext, ok_a, err_a = _transform_extended(view_a, ext_a, len(declared_a))
    oh_b_ext, ok_b, err_b = _transform_extended(view_b, ext_b, len(declared_b))
    z_a_ext = _sum_matrix(encoder_a, field_a, names_a, ext_a)
    z_b_ext = _sum_matrix(encoder_b, field_b, names_b, ext_b)

    old_grid = (
        old_intercept
        + (oh_a_ext @ w_a)[:, None]
        + (oh_b_ext @ w_b)[None, :]
        + oh_a_ext @ mat_interaction @ oh_b_ext.T
    )
    new_grid = (
        new_intercept
        + (z_a_ext @ new_w_a)[:, None]
        + (z_b_ext @ new_w_b)[None, :]
        + z_a_ext @ new_interaction @ z_b_ext.T
    )

    notes: list[str] = []
    for field_name, values, _ok, errors in (
        (field_a, ext_a, ok_a, err_a),
        (field_b, ext_b, ok_b, err_b),
    ):
        for i, exc in errors.items():
            notes.append(
                f'{field_name}={values[i]!r}: the old encoder cannot score this input '
                f'({type(exc).__name__}: {exc}); grid cells involving it were skipped.'
            )

    scorable = ok_a[:, None] & ok_b[None, :]
    diff = np.abs(old_grid - new_grid)
    tolerance = atol + rtol * np.abs(old_grid)
    n_declared_cells = len(declared_a) * len(declared_b)
    declared_block = np.zeros_like(scorable, dtype=bool)
    declared_block[: len(declared_a), : len(declared_b)] = True
    max_diff_declared = float(diff[declared_block & scorable].max(initial=0.0))
    max_diff = float(diff[scorable].max(initial=0.0))

    bad_rows, bad_cols = np.nonzero(scorable & (diff > tolerance))
    truncated = False
    for i, j in zip(bad_rows, bad_cols, strict=True):
        if len(conflicts) >= _MAX_STORED_CONFLICTS:
            truncated = True
            break
        kind = _cell_kind(kinds_a[i], kinds_b[j])
        parts = []
        for kind_, value, fname, view, old_row, new_row in (
            (kinds_a[i], ext_a[i], field_a, view_a, oh_a_ext[i], z_a_ext[i]),
            (kinds_b[j], ext_b[j], field_b, view_b, oh_b_ext[j], z_b_ext[j]),
        ):
            if kind_ == _DECLARED:
                continue
            parts.append(
                f'{fname}={value!r} ({kind_}): old {view.desc} encodes it as '
                f'{_row_desc(old_row)}, the new SumEncoder encodes it as '
                f'{_row_desc(new_row)}'
            )
        detail = '; '.join(parts) if parts else 'declared-category combination'
        conflicts.append(
            MigrationConflict(
                field_a=field_a,
                field_b=field_b,
                value_a=ext_a[i],
                value_b=ext_b[j],
                kind=kind,
                old_score=float(old_grid[i, j]),
                new_score=float(new_grid[i, j]),
                abs_diff=float(diff[i, j]),
                reason=(
                    f'{detail}; old score {old_grid[i, j]:.6g} vs new score '
                    f'{new_grid[i, j]:.6g} (|diff| {diff[i, j]:.3g})'
                ),
            )
        )
    if truncated:
        notes.append(
            f'Only the first {_MAX_STORED_CONFLICTS} conflicts are stored; '
            'more combinations differ.'
        )

    n_unscorable = int((~scorable).sum())
    report = SumMigrationReport(
        n_declared_cells=n_declared_cells,
        n_checked_cells=int(scorable.sum()),
        n_unscorable_cells=n_unscorable,
        max_abs_diff_declared=max_diff_declared,
        max_abs_diff=max_diff,
        conflicts=conflicts,
        notes=notes,
    )
    result = SumMigrationResult(
        encoder_a=encoder_a,
        encoder_b=encoder_b,
        field_a=field_a,
        field_b=field_b,
        intercept=new_intercept,
        main_effects_a=new_w_a,
        main_effects_b=new_w_b,
        interaction=new_interaction,
        feature_names_a=names_a,
        feature_names_b=names_b,
        declared_categories_a=list(declared_a),
        declared_categories_b=list(declared_b),
        report=report,
    )

    if conflicts and on_conflict == 'raise':
        lines = [
            f'Cannot re-express the old linear score of fields {field_a!r} and '
            f'{field_b!r} in SumEncoder space exactly: '
            f'{len(conflicts)} of {report.n_checked_cells} checked input '
            'combinations have different old/new scores.'
        ]
        if all(c.kind != 'unrepresentable' for c in conflicts):
            lines.append(
                f'All {report.n_declared_cells} declared-category combinations match '
                f'(max |diff| {report.max_abs_diff_declared:.3g}); the conflicts '
                'involve missing or unseen inputs.'
            )
        shown = conflicts[:5]
        lines.extend(f'  - {c.reason}' for c in shown)
        if len(conflicts) > len(shown):
            lines.append(f'  ... and {len(conflicts) - len(shown)} more.')
        lines.append(
            "Call with on_conflict='return' to get the migrated coefficients "
            'together with the full conflict report.'
        )
        raise SumMigrationConflictError('\n'.join(lines), conflicts)

    return result
