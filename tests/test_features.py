import pandas as pd

from lol_win.features.build import clip_upper, side_diffs


def test_side_diffs_blue_minus_red() -> None:
    df = pd.DataFrame({"blue_kills": [5, 2], "red_kills": [3, 4]})
    assert side_diffs(df, ["kills"])["kills_diff"].tolist() == [2, -2]


def test_side_diffs_symmetric_when_swapping_sides() -> None:
    df = pd.DataFrame({"blue_kills": [5, 2], "red_kills": [3, 4]})
    swapped = df.rename(columns={"blue_kills": "red_kills", "red_kills": "blue_kills"})
    pd.testing.assert_frame_equal(side_diffs(swapped, ["kills"]), -side_diffs(df, ["kills"]))


def test_clip_upper_caps_only_above_threshold() -> None:
    assert clip_upper(pd.Series([1, 50, 200]), upper=100).tolist() == [1, 50, 100]
