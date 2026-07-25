"""Fail-closed adapter for FastPLAID's immutable premerged runtime format."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np


FAST_PLAID_VERSION = "1.3.0.290"
_SMALL_TENSORS = (
    "centroids.npy",
    "avg_residual.npy",
    "bucket_cutoffs.npy",
    "bucket_weights.npy",
    "ivf.npy",
    "ivf_lengths.npy",
)
_MERGED_TENSOR_DTYPES = {
    "merged_codes.npy": np.dtype(np.int64),
    "merged_residuals.npy": np.dtype(np.uint8),
}
_TENSOR_NAMES = (*_SMALL_TENSORS, *_MERGED_TENSOR_DTYPES)


def _plain_shape(array: np.ndarray) -> list[int]:
    return [int(value) for value in array.shape]


def inspect_immutable_premerged_artifacts(
    index_path: Path,
    *,
    expected_document_count: int,
    expected_embedding_dimension: int,
) -> dict[str, Any]:
    """Validate and describe the exact artifacts consumed by production."""
    root = index_path.resolve()
    metadata_path = index_path / "metadata.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing immutable FastPLAID artifact: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read FastPLAID metadata: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("FastPLAID metadata must be a JSON object")
    num_chunks = metadata.get("num_chunks")
    nbits = metadata.get("nbits")
    if (
        not isinstance(num_chunks, int)
        or isinstance(num_chunks, bool)
        or num_chunks < 1
    ):
        raise ValueError("FastPLAID metadata num_chunks must be positive")
    if not isinstance(nbits, int) or isinstance(nbits, bool) or nbits < 1:
        raise ValueError("FastPLAID metadata nbits must be positive")

    artifacts: dict[str, dict[str, Any]] = {
        "metadata.json": {"size_bytes": metadata_path.stat().st_size}
    }
    doc_lengths: list[int] = []
    for chunk in range(num_chunks):
        relative = f"doclens.{chunk}.json"
        path = index_path / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing immutable FastPLAID artifact: {path}")
        try:
            chunk_lengths = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read FastPLAID document lengths: {exc}") from exc
        if (
            not isinstance(chunk_lengths, list)
            or any(
                not isinstance(item, int)
                or isinstance(item, bool)
                or item < 1
                for item in chunk_lengths
            )
        ):
            raise ValueError(f"FastPLAID document lengths are invalid: {path}")
        doc_lengths.extend(chunk_lengths)
        artifacts[relative] = {"size_bytes": path.stat().st_size}
    if len(doc_lengths) != expected_document_count:
        raise ValueError(
            "FastPLAID document length count does not match corpus_count"
        )

    arrays: dict[str, np.ndarray] = {}
    try:
        for relative in _TENSOR_NAMES:
            path = index_path / relative
            try:
                path.resolve().relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"FastPLAID artifact escapes its directory: {path}"
                ) from exc
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(
                    f"missing immutable FastPLAID artifact: {path}"
                )
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"cannot map immutable FastPLAID artifact {path}: {exc}"
                ) from exc
            expected_dtype = _MERGED_TENSOR_DTYPES.get(relative)
            if expected_dtype is not None and array.dtype != expected_dtype:
                raise ValueError(
                    f"FastPLAID artifact {relative} has dtype {array.dtype}, "
                    f"expected {expected_dtype}"
                )
            if array.dtype.hasobject:
                raise ValueError(f"FastPLAID artifact has object dtype: {path}")
            if array.size < 1:
                raise ValueError(f"FastPLAID artifact is empty: {path}")
            arrays[relative] = array
            artifacts[relative] = {
                "dtype": array.dtype.name,
                "shape": _plain_shape(array),
                "size_bytes": path.stat().st_size,
            }

        centroids = arrays["centroids.npy"]
        codes = arrays["merged_codes.npy"]
        residuals = arrays["merged_residuals.npy"]
        if centroids.ndim != 2 or centroids.shape[1] != expected_embedding_dimension:
            raise ValueError(
                "FastPLAID centroids do not match build.embedding_dimension"
            )
        if codes.ndim != 1 or residuals.ndim != 2:
            raise ValueError("FastPLAID merged tensor ranks are incompatible")
        padding_rows = max(0, max(doc_lengths) - doc_lengths[-1])
        expected_rows = sum(doc_lengths) + padding_rows
        if codes.shape[0] != expected_rows or residuals.shape[0] != expected_rows:
            raise ValueError(
                "FastPLAID merged tensors are not aligned with document lengths"
            )
    finally:
        for array in arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    return {
        "schema": 1,
        "fast_plaid_version": FAST_PLAID_VERSION,
        "num_chunks": num_chunks,
        "nbits": nbits,
        "document_count": len(doc_lengths),
        "padding_rows": padding_rows,
        "artifacts": artifacts,
    }


@dataclass(frozen=True)
class _Dependencies:
    torch: Any
    fast_plaid_rust: Any
    fast_plaid_class: type
    load_torch_path: Any
    construct_index_from_tensors: Any


def _load_dependencies() -> _Dependencies:
    """Keep all private FastPLAID imports behind this version gate."""
    try:
        installed = version("fast-plaid")
    except PackageNotFoundError as exc:
        raise RuntimeError("FastPLAID is required for immutable production search") from exc
    if installed != FAST_PLAID_VERSION:
        raise RuntimeError(
            f"immutable FastPLAID adapter requires fast-plaid=={FAST_PLAID_VERSION}; "
            f"found {installed}"
        )

    import torch
    from fast_plaid import fast_plaid_rust
    from fast_plaid.search.fast_plaid import FastPlaid, _load_torch_path
    from fast_plaid.search.load import _construct_index_from_tensors

    return _Dependencies(
        torch=torch,
        fast_plaid_rust=fast_plaid_rust,
        fast_plaid_class=FastPlaid,
        load_torch_path=_load_torch_path,
        construct_index_from_tensors=_construct_index_from_tensors,
    )


class ImmutablePremergedFastPlaid:
    """FastPLAID search facade retaining its read-only mapped CPU tensors."""

    def __init__(
        self,
        backend: Any,
        arrays: list[np.ndarray],
        cpu_tensors: dict[str, Any],
    ) -> None:
        self._backend = backend
        self._arrays = arrays
        self._cpu_tensors = cpu_tensors

    def search(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend.search(*args, **kwargs)

    def close(self) -> None:
        self._backend = None
        self._cpu_tensors.clear()
        for array in self._arrays:
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._arrays.clear()


def load_immutable_premerged_fast_plaid(
    index_path: Path,
    certificate: dict[str, Any],
    *,
    _dependencies: _Dependencies | None = None,
) -> ImmutablePremergedFastPlaid:
    """Construct FastPLAID directly from certified, read-only merged tensors."""
    expected_count = certificate.get("document_count")
    centroids_certificate = certificate.get("artifacts", {}).get("centroids.npy", {})
    centroid_shape = centroids_certificate.get("shape")
    if (
        not isinstance(expected_count, int)
        or not isinstance(centroid_shape, list)
        or len(centroid_shape) != 2
        or not isinstance(centroid_shape[1], int)
    ):
        raise ValueError("immutable premerged certificate is incompatible")
    inspected = inspect_immutable_premerged_artifacts(
        index_path,
        expected_document_count=expected_count,
        expected_embedding_dimension=centroid_shape[1],
    )
    if inspected != certificate:
        raise ValueError("immutable premerged artifacts do not match their certificate")

    dependencies = _dependencies or _load_dependencies()
    torch = dependencies.torch
    arrays: list[np.ndarray] = []
    try:
        def mapped(relative: str) -> np.ndarray:
            array = np.load(
                index_path / relative,
                mmap_mode="r",
                allow_pickle=False,
            )
            if array.flags.writeable:
                raise RuntimeError(f"FastPLAID artifact was not mapped read-only: {relative}")
            arrays.append(array)
            return array

        cpu_tensors = {
            "nbits": certificate["nbits"],
            "centroids": torch.from_numpy(mapped("centroids.npy")).to(
                device="cpu", dtype=torch.float16
            ),
            "avg_residual": torch.from_numpy(mapped("avg_residual.npy")).to(
                device="cpu", dtype=torch.float16
            ),
            "bucket_cutoffs": torch.from_numpy(mapped("bucket_cutoffs.npy")).to(
                device="cpu", dtype=torch.float16
            ),
            "bucket_weights": torch.from_numpy(mapped("bucket_weights.npy")).to(
                device="cpu", dtype=torch.float16
            ),
            "ivf": torch.from_numpy(mapped("ivf.npy")).to(
                device="cpu", dtype=torch.int64
            ),
            "ivf_lengths": torch.from_numpy(mapped("ivf_lengths.npy")).to(
                device="cpu", dtype=torch.int32
            ),
            "doc_codes": torch.from_numpy(mapped("merged_codes.npy")),
            "doc_residuals": torch.from_numpy(mapped("merged_residuals.npy")),
        }
        all_doc_lengths: list[int] = []
        for chunk in range(certificate["num_chunks"]):
            all_doc_lengths.extend(
                json.loads(
                    (index_path / f"doclens.{chunk}.json").read_text(encoding="utf-8")
                )
            )
        cpu_tensors["doc_lengths"] = torch.tensor(
            all_doc_lengths,
            device="cpu",
            dtype=torch.int64,
        )

        if torch.cuda.is_available():
            devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        else:
            devices = ["cpu"]
        devices = list(dict.fromkeys(devices))
        torch_path = dependencies.load_torch_path(device=devices[0])
        dependencies.fast_plaid_rust.initialize_torch(torch_path=torch_path)

        def provision(device: str) -> tuple[str, Any]:
            index = dependencies.construct_index_from_tensors(cpu_tensors, device)
            if index is None:
                raise RuntimeError(f"FastPLAID failed to construct index on {device}")
            return device, index

        if len(devices) == 1:
            indices = dict([provision(devices[0])])
        else:
            with ThreadPoolExecutor(max_workers=len(devices)) as executor:
                indices = dict(executor.map(provision, devices))

        backend = dependencies.fast_plaid_class.__new__(
            dependencies.fast_plaid_class
        )
        backend.devices = devices
        backend.torch_path = torch_path
        backend.index = str(index_path)
        backend.indices = indices
        return ImmutablePremergedFastPlaid(backend, arrays, cpu_tensors)
    except BaseException:
        for array in arrays:
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        raise
