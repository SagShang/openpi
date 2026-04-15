import pathlib

import pytest

import openpi.shared.download as download


@pytest.fixture(scope="session", autouse=True)
def set_openpi_data_home(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("openpi_data")
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("OPENPI_DATA_HOME", str(temp_dir))
        yield


def test_download_local(tmp_path: pathlib.Path):
    local_path = tmp_path / "local"
    local_path.touch()

    result = download.maybe_download(str(local_path))
    assert result == local_path

    with pytest.raises(FileNotFoundError):
        download.maybe_download("bogus")


def test_download_gs_dir():
    remote_path = "gs://openpi-assets/testdata/random"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_gs():
    remote_path = "gs://openpi-assets/testdata/random/random_512kb.bin"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_fsspec():
    remote_path = "gs://big_vision/paligemma_tokenizer.model"

    local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert new_local_path == local_path


def test_download_uses_symlinked_cache_path():
    cache_dir = download.get_cache_dir()
    target_dir = cache_dir.parent / "symlink_target"
    target_dir.mkdir(exist_ok=True)

    linked_parent = cache_dir / "openpi-assets"
    linked_parent.mkdir(exist_ok=True)
    (linked_parent / "checkpoints").symlink_to(target_dir, target_is_directory=True)

    cached_path = linked_parent / "checkpoints" / "pi05_base" / "assets" / "franka"
    cached_path.mkdir(parents=True)
    (cached_path / "norm_stats.json").write_text("{}")

    result = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base/assets/franka")

    assert result == cached_path
    assert result.exists()
