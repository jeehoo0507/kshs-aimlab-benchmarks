"""Small archive fixtures catch label-order and archive-path mistakes."""

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest


PREPARE = Path(__file__).resolve().parents[1] / "datasets/imagenet/prepare.py"


@pytest.fixture(scope="module")
def prepare_module():
    pytest.importorskip("scipy")
    spec = importlib.util.spec_from_file_location("imagenet_prepare", PREPARE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive_bytes(members):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_train_and_val_preparation_preserves_official_label_order(tmp_path, prepare_module):
    mapping = {1: "n00000001", 2: "n00000002"}
    val = archive_bytes([
        (f"ILSVRC2012_val_{number:08d}.JPEG", bytes([number]))
        for number in range(1, 5)
    ])
    val_path = tmp_path / "val.tar"
    val_path.write_bytes(val)
    prepare_module.prepare_val(val_path, tmp_path, mapping, [2, 1, 2, 1])
    assert (tmp_path / "val/n00000002/ILSVRC2012_val_00000001.JPEG").read_bytes() == b"\x01"
    assert (tmp_path / "val/n00000001/ILSVRC2012_val_00000002.JPEG").read_bytes() == b"\x02"

    train = archive_bytes([
        (f"{wnid}.tar", archive_bytes([(f"{wnid}_1.JPEG", wnid.encode())]))
        for wnid in mapping.values()
    ])
    train_path = tmp_path / "train.tar"
    train_path.write_bytes(train)
    prepare_module.prepare_train(train_path, tmp_path, set(mapping.values()))
    for wnid in mapping.values():
        assert (tmp_path / "train" / wnid / f"{wnid}_1.JPEG").read_bytes() == wnid.encode()


def test_validation_rejects_archive_path_traversal(tmp_path, prepare_module):
    path = tmp_path / "invalid.tar"
    path.write_bytes(archive_bytes([("../escape.JPEG", b"invalid")]))
    with pytest.raises(ValueError, match="Unexpected validation archive member"):
        prepare_module.prepare_val(path, tmp_path, {1: "n00000001"}, [1])
    assert not (tmp_path / "escape.JPEG").exists()


def test_devkit_maps_validation_ids_to_wnids(tmp_path, prepare_module):
    import numpy as np
    from scipy.io import savemat

    synsets = np.empty((1000, 5), dtype=object)
    for index in range(1000):
        synsets[index] = (index + 1, f"n{index + 1:08d}", "name", "description", 0)
    mat = io.BytesIO()
    savemat(mat, {"synsets": synsets})
    data = archive_bytes([
        ("ILSVRC2012_devkit_t12/data/meta.mat", mat.getvalue()),
        ("ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
         b"2\n" + b"1\n" * 49999),
    ])
    path = tmp_path / "devkit.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as source:
            for member in source:
                payload = source.extractfile(member).read()
                archive.addfile(member, io.BytesIO(payload))
    mapping, labels = prepare_module.read_devkit(path)
    assert mapping[2] == "n00000002"
    assert labels[:2] == [2, 1]
    assert len(labels) == 50000
