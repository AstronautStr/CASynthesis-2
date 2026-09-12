"""Snapshot files (S5): a state tree = JSON scalars / lists / dicts + numpy
arrays, stored as <name>.json + <name>.npz side by side.

JSON keeps the structure and the scalars; every ndarray leaf is replaced by
{"__array__": "<key>"} and stored under that key in the npz (dtype and
precision preserved).  Loading never unpickles (allow_pickle=False) and checks
that every referenced array exists.  No objects other than these types are
accepted -- a state must be EXPLICIT, not a dump of live objects.
"""
import json
import math
import os

import numpy as np

SNAPSHOT_FILE_VERSION = 1


class SnapshotError(ValueError):
    pass


def _to_json(node, arrays, path):
    if isinstance(node, np.ndarray):
        key = path.strip('/').replace('/', '.') or 'root'
        if node.dtype == object:
            raise SnapshotError(f"{path}: object array cannot be stored")
        arrays[key] = node
        return {"__array__": key}
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if not isinstance(k, str):
                raise SnapshotError(f"{path}: key {k!r} is not a string")
            out[k] = _to_json(v, arrays, f"{path}/{k}")
        return out
    if isinstance(node, (list, tuple)):
        return [_to_json(v, arrays, f"{path}/{i}") for i, v in enumerate(node)]
    if isinstance(node, np.integer):
        return int(node)
    if isinstance(node, np.floating):
        node = float(node)
    if isinstance(node, bool) or node is None or isinstance(node, (int, str)):
        return node
    if isinstance(node, float):
        if not math.isfinite(node):
            raise SnapshotError(f"{path}: non-finite float")
        return node
    raise SnapshotError(f"{path}: unsupported value type {type(node).__name__}")


def _from_json(node, arrays, path):
    if isinstance(node, dict):
        if set(node) == {"__array__"}:
            key = node["__array__"]
            if key not in arrays:
                raise SnapshotError(f"{path}: array {key!r} missing from the npz")
            return arrays[key]
        return {k: _from_json(v, arrays, f"{path}/{k}") for k, v in node.items()}
    if isinstance(node, list):
        return [_from_json(v, arrays, f"{path}/{i}") for i, v in enumerate(node)]
    return node


def pack(state):
    """state tree -> (json document, {key: ndarray})."""
    arrays = {}
    doc = dict(snapshot_file_version=SNAPSHOT_FILE_VERSION,
               state=_to_json(state, arrays, ''))
    return doc, arrays


def unpack(doc, arrays):
    ver = doc.get('snapshot_file_version') if isinstance(doc, dict) else doc
    if not isinstance(doc, dict) or ver != SNAPSHOT_FILE_VERSION:
        raise SnapshotError(f"snapshot file version {ver!r} != {SNAPSHOT_FILE_VERSION}")
    if 'state' not in doc:
        raise SnapshotError("snapshot file: missing 'state'")
    return _from_json(doc['state'], arrays, '')


def save_state(directory, name, state):
    """Write <name>.json + <name>.npz; returns the two file names."""
    doc, arrays = pack(state)
    jp, ap = os.path.join(directory, name + '.json'), os.path.join(directory, name + '.npz')
    with open(jp, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    with open(ap, 'wb') as f:
        np.savez(f, **{k: np.ascontiguousarray(v) for k, v in arrays.items()})
    return name + '.json', name + '.npz'


def load_state(directory, name):
    jp, ap = os.path.join(directory, name + '.json'), os.path.join(directory, name + '.npz')
    try:
        with open(jp, encoding='utf-8') as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SnapshotError(f"{name}.json: {e}")
    try:
        with np.load(ap, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    except Exception as e:                 # noqa: BLE001  (BadZipFile, EOFError, OSError...)
        raise SnapshotError(f"{name}.npz: {e}")
    return unpack(doc, arrays)


def copy_state(state):
    """Deep copy through pack/unpack (arrays copied) -- a pristine origin."""
    doc, arrays = pack(state)
    return unpack(json.loads(json.dumps(doc)), {k: v.copy() for k, v in arrays.items()})
