import lol_win
from lol_win import config


def test_package_imports() -> None:
    assert lol_win.__version__


def test_random_state_is_fixed() -> None:
    assert config.RANDOM_STATE == 42


def test_project_root_points_to_repo() -> None:
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()


def test_raw_csv_exists() -> None:
    assert config.RAW_CSV.is_file()
