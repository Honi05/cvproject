from pathlib import Path
import cvchess.data.download as dl


def test_is_cached_false_when_missing(tmp_path):
    assert dl.is_cached(tmp_path / "nope") is False


def test_is_cached_true_when_marker_present(tmp_path):
    d = tmp_path / "ds"
    (d / "train").mkdir(parents=True)
    (d / "train" / "a.png").write_bytes(b"x")
    (d / "test").mkdir()
    (d / "test" / "b.png").write_bytes(b"x")
    assert dl.is_cached(d) is True


def test_download_skips_when_cached(tmp_path, monkeypatch):
    d = tmp_path / "ds"
    (d / "train").mkdir(parents=True)
    (d / "train" / "a.png").write_bytes(b"x")
    (d / "test").mkdir()
    (d / "test" / "b.png").write_bytes(b"x")
    called = {"n": 0}
    def fake_dl(slug):
        called["n"] += 1
        return str(d)
    monkeypatch.setattr(dl.kagglehub, "dataset_download", fake_dl)
    out = dl.ensure_dataset(target_dir=d)
    assert out == d
    assert called["n"] == 0
