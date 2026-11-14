import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from lol_win.config import INPUT_COLUMNS, SIDES, TEAM_STATS
from lol_win.features.build import (
    FeatureBuilder,
    WardsClipper,
    build_features,
    clip_upper,
    make_feature_pipeline,
    side_diffs,
)


def _random_stats(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.integers(0, 30, n) for col in INPUT_COLUMNS}
    data["blue_first_blood"] = rng.integers(0, 2, n)
    for side in SIDES:
        data[f"{side}_avg_level"] = rng.integers(30, 40, n) / 5
    return pd.DataFrame(data)


def _swap_sides(stats: pd.DataFrame) -> pd.DataFrame:
    swapped = stats.rename(
        columns={
            **{f"blue_{s}": f"red_{s}" for s in TEAM_STATS},
            **{f"red_{s}": f"blue_{s}" for s in TEAM_STATS},
        }
    )
    return swapped.assign(blue_first_blood=1 - stats["blue_first_blood"])


def test_side_diffs_blue_minus_red() -> None:
    df = pd.DataFrame({"blue_kills": [5, 2], "red_kills": [3, 4]})
    assert side_diffs(df, ["kills"])["kills_diff"].tolist() == [2, -2]


def test_clip_upper_caps_only_above_threshold() -> None:
    assert clip_upper(pd.Series([1, 50, 200]), upper=100).tolist() == [1, 50, 100]


def test_build_features_diffs_columns() -> None:
    features = build_features(_random_stats())
    expected = [f"{s}_diff" for s in TEAM_STATS] + ["blue_first_blood"]
    assert features.columns.tolist() == expected


def test_build_features_diffs_abs_adds_absolutes() -> None:
    stats = _random_stats()
    features = build_features(stats, "diffs_abs")
    assert features.shape[1] == len(TEAM_STATS) + 1 + 2 * len(TEAM_STATS)
    assert (features["blue_total_gold"] == stats["blue_total_gold"]).all()


def test_build_features_is_symmetric_when_swapping_sides() -> None:
    stats = _random_stats()
    original = build_features(stats)
    swapped = build_features(_swap_sides(stats))
    diff_cols = [f"{s}_diff" for s in TEAM_STATS]
    pd.testing.assert_frame_equal(swapped[diff_cols], -original[diff_cols])
    pd.testing.assert_series_equal(swapped["blue_first_blood"], 1 - original["blue_first_blood"])


def test_build_features_ignores_extra_columns() -> None:
    stats = _random_stats().assign(game_id=1, blue_wins=0)
    assert "game_id" not in build_features(stats).columns


def test_build_features_rejects_unknown_set_and_missing_columns() -> None:
    stats = _random_stats()
    with pytest.raises(ValueError, match="feature_set"):
        build_features(stats, "everything")  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="blue_total_gold"):
        build_features(stats.drop(columns="blue_total_gold"))


def test_wards_clipper_learns_cap_only_on_fit() -> None:
    train = _random_stats().assign(blue_wards_placed=10, red_wards_placed=10)
    test = _random_stats().assign(blue_wards_placed=500, red_wards_placed=3)
    clipper = WardsClipper(quantile=0.99).fit(train)
    transformed = clipper.transform(test)
    assert clipper.cap_ == 10
    assert (transformed["blue_wards_placed"] == 10).all()
    assert (transformed["red_wards_placed"] == 3).all()
    assert (test["blue_wards_placed"] == 500).all(), "transform não pode alterar a entrada"


def test_wards_clipper_uses_same_cap_for_both_sides() -> None:
    stats = _random_stats(n=200)
    clipper = WardsClipper(quantile=0.9).fit(stats)
    pooled = pd.concat([stats["blue_wards_placed"], stats["red_wards_placed"]])
    assert clipper.cap_ == pytest.approx(pooled.quantile(0.9))
    transformed = clipper.transform(stats)
    assert transformed[["blue_wards_placed", "red_wards_placed"]].max().max() <= clipper.cap_


def test_wards_clipper_none_is_passthrough() -> None:
    stats = _random_stats().assign(blue_wards_placed=999)
    out = WardsClipper(quantile=None).fit(stats).transform(stats)
    pd.testing.assert_frame_equal(out, stats)


def test_transformers_require_fit() -> None:
    with pytest.raises(NotFittedError):
        WardsClipper().transform(_random_stats())
    with pytest.raises(NotFittedError):
        FeatureBuilder().transform(_random_stats())


def test_feature_pipeline_end_to_end_and_clonable() -> None:
    stats = _random_stats()
    pipeline = make_feature_pipeline("diffs", wards_quantile=0.99)
    features = pipeline.fit_transform(stats)
    assert features.columns.tolist() == pipeline.get_feature_names_out().tolist()
    cloned = clone(pipeline).set_params(
        features__feature_set="diffs_abs", clip_wards__quantile=None
    )
    assert cloned.fit_transform(stats).shape[1] == 3 * len(TEAM_STATS) + 1
