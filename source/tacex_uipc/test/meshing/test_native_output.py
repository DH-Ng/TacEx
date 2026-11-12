"""Unit tests for the isolation of native console output / working directory files (``utils/native_output.py``).

Pure python test (+ wildmeshing), no Isaac Sim app needed:
    python -m pytest source/tacex_uipc/test/meshing/test_native_output.py -v
"""

import ctypes
import importlib.util
import os
import pathlib

import numpy as np
import pytest

# load the module directly: importing the `tacex_uipc` package pulls in Isaac Sim
_spec = importlib.util.spec_from_file_location(
    "native_output",
    pathlib.Path(__file__).resolve().parents[2] / "tacex_uipc" / "utils" / "native_output.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
isolate_native_output = _mod.isolate_native_output

_LIBC = ctypes.CDLL(None)


def _native_print(text: str):
    """Writes via C stdio (like C++ std::cout), bypassing Python's sys.stdout."""
    _LIBC.puts(text.encode())
    _LIBC.fflush(None)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    d = tmp_path / "cwd"
    d.mkdir()
    monkeypatch.chdir(d)
    return d


def test_native_output_and_files_isolated(workdir, capfd):
    with isolate_native_output("probe") as tmp_dir:
        _native_print("native stdout")
        os.write(2, b"native stderr\n")
        pathlib.Path("__tracked_surface.stl").write_text("debug")
        assert os.getcwd() == tmp_dir
    out, err = capfd.readouterr()
    # (Python prints are not checked: pytest replaces sys.stdout, so they bypass fd 1 only under pytest)
    assert "native" not in out + err
    assert os.getcwd() == str(workdir)
    assert list(workdir.iterdir()) == []  # nothing written into the real working directory
    assert not os.path.exists(tmp_dir)  # temporary directory removed
    # console works again afterwards
    _native_print("after")
    assert "after" in capfd.readouterr().out


def test_log_kept_in_log_dir(workdir, tmp_path):
    log_dir = tmp_path / "logs"
    with isolate_native_output("probe", log_dir=str(log_dir)):
        _native_print("keep me")
    logs = list(log_dir.glob("probe_*.log"))
    assert len(logs) == 1 and "keep me" in logs[0].read_text()


def test_restored_on_error(workdir, capfd):
    with pytest.raises(RuntimeError):
        with isolate_native_output("probe"):
            _native_print("line before failure")
            raise RuntimeError("boom")
    assert os.getcwd() == str(workdir)
    _, err = capfd.readouterr()
    assert "line before failure" in err  # log tail shown for debugging


def test_wildmeshing_isolated(workdir, capfd):
    wm = pytest.importorskip("wildmeshing")
    c = np.array([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
    t = np.array([0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4,
                  1, 5, 7, 1, 7, 3], dtype=np.uint32).reshape(-1, 3)  # fmt: skip
    with isolate_native_output("wildmeshing"):
        tet = wm.Tetrahedralizer(stop_quality=8, max_its=20, epsilon=0.01, edge_length_r=0.5, coarsen=True)
        tet.set_log_level(6)
        tet.set_mesh(c, t)
        tet.tetrahedralize()
        points, tets, _ = tet.get_tet_mesh(correct_surface_orientation=True, manifold_surface=True)
    out, err = capfd.readouterr()
    assert out == "" and err == ""
    assert list(workdir.iterdir()) == []  # no __tracked_surface.stl
    assert np.asarray(points).shape[0] > 0 and np.asarray(tets).shape[1] == 4
