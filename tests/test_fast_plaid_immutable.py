from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from agentkb.fast_plaid_immutable import (
    _Dependencies,
    inspect_immutable_premerged_artifacts,
    load_immutable_premerged_fast_plaid,
)


def _write_artifacts(path):
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(json.dumps({"num_chunks": 1, "nbits": 4}))
    (path / "doclens.0.json").write_text("[2, 1]")
    np.save(path / "centroids.npy", np.ones((2, 4), dtype=np.float16))
    np.save(path / "avg_residual.npy", np.ones(1, dtype=np.float32))
    np.save(path / "bucket_cutoffs.npy", np.ones(2, dtype=np.float32))
    np.save(path / "bucket_weights.npy", np.ones(2, dtype=np.float32))
    np.save(path / "ivf.npy", np.ones(2, dtype=np.int64))
    np.save(path / "ivf_lengths.npy", np.ones(2, dtype=np.int32))
    np.save(path / "merged_codes.npy", np.arange(4, dtype=np.int64))
    np.save(path / "merged_residuals.npy", np.ones((4, 2), dtype=np.uint8))
    return inspect_immutable_premerged_artifacts(
        path,
        expected_document_count=2,
        expected_embedding_dimension=4,
    )


def test_missing_merged_artifact_fails_closed(tmp_path):
    certificate = _write_artifacts(tmp_path / "fast_plaid_index")
    (tmp_path / "fast_plaid_index" / "merged_codes.npy").unlink()

    with pytest.raises(FileNotFoundError, match="merged_codes"):
        load_immutable_premerged_fast_plaid(
            tmp_path / "fast_plaid_index",
            certificate,
        )


def test_constructs_from_aligned_read_only_merged_tensors_and_closes(tmp_path):
    index_path = tmp_path / "fast_plaid_index"
    certificate = _write_artifacts(index_path)
    constructed = []

    class FakeTensor:
        def __init__(self, array):
            self.array = array

        def to(self, **_kwargs):
            return self

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

    class FakeTorch:
        float16 = "float16"
        int64 = "int64"
        int32 = "int32"
        cuda = FakeCuda()

        @staticmethod
        def from_numpy(array):
            return FakeTensor(array)

        @staticmethod
        def tensor(values, **_kwargs):
            return FakeTensor(np.asarray(values))

    class FakeBackend:
        def search(self, *_args, **_kwargs):
            return []

    def construct(tensors, device):
        assert device == "cpu"
        assert tensors["doc_codes"].array.shape == (4,)
        assert tensors["doc_residuals"].array.shape == (4, 2)
        assert not tensors["doc_codes"].array.flags.writeable
        assert not tensors["doc_residuals"].array.flags.writeable
        constructed.append(tensors)
        return object()

    initialized = []
    dependencies = _Dependencies(
        torch=FakeTorch,
        fast_plaid_rust=SimpleNamespace(
            initialize_torch=lambda **kwargs: initialized.append(kwargs)
        ),
        fast_plaid_class=FakeBackend,
        load_torch_path=lambda **_kwargs: "/torch",
        construct_index_from_tensors=construct,
    )
    backend = load_immutable_premerged_fast_plaid(
        index_path,
        certificate,
        _dependencies=dependencies,
    )

    assert initialized == [{"torch_path": "/torch"}]
    assert len(constructed) == 1
    mapped_arrays = list(backend._arrays)
    backend.close()
    assert backend._arrays == []
    assert backend._cpu_tensors == {}
    assert all(array._mmap.closed for array in mapped_arrays)
