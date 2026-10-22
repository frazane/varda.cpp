"""Read a pickled PyTorch checkpoint without the code that produced it.

Written for anemoi checkpoints (a pickled `AnemoiModelInterface`), but model-agnostic.
We unpickle with a restricted unpickler: torch tensor/storage rebuild functions, dtypes,
numpy array reconstruction and a few builtins are real; every other global (the model
library, torch.nn, torch_geometric, omegaconf, ...) becomes an inert `Stub` that only
records its qualified name, constructor args, state and items. No foreign code runs.

    root = load("inference-last.ckpt")
    tensors = flat_tensors(root)        # {dotted.path: tensor}, params + buffers + attrs
    for path, cls, attrs in modules(root): ...
    to_plain(stub)                      # non-tensor config -> plain dict/list values

Run `.venv/bin/python -I python/checkpoint.py CKPT` for a printed inventory.
"""

import _compat_pickle
import collections
import hashlib
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------- stubs


class Stub:
    """Inert stand-in for a foreign class or instance. Never runs foreign code."""

    _qualname = "?"

    def __new__(cls, *args, **kwargs):  # NEWOBJ / REDUCE
        self = object.__new__(cls)
        self.__dict__.update(_stub_args=args, _stub_kwargs=kwargs, _stub_items={}, _stub_list=[], _stub_state=None)
        return self

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):  # BUILD
        slots = None
        if isinstance(state, tuple) and len(state) == 2 and all(s is None or isinstance(s, dict) for s in state):
            state, slots = state  # (dict, slotstate) convention
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_stub_state"] = state
        if slots:
            self.__dict__.update(slots)

    def __setitem__(self, key, value):  # SETITEM(S) on dict subclasses
        self._stub_items[key] = value

    def append(self, value):  # APPEND(S) on list subclasses
        self._stub_list.append(value)

    def extend(self, values):
        self._stub_list.extend(values)

    def __repr__(self):
        return f"<{self._qualname}>"


_stub_classes = {}


def stub_class(qualname):
    """One Stub subclass per qualified name, so instances keep their class identity."""
    if qualname not in _stub_classes:
        module, _, name = qualname.rpartition(".")
        ns = {"_qualname": qualname, "__module__": module, "__qualname__": name}
        _stub_classes[qualname] = type(name, (Stub,), ns)
    return _stub_classes[qualname]


def _safe_getattr(obj, name):
    """`builtins.getattr` replacement: only resolves attributes of stubs, to stubs.

    getattr(StubClass, "apply") -> stub class `<qualname>.apply`;
    getattr(stub_instance, "forward") -> stub instance `<qualname>.forward` with args (instance,)
    (a bound method reference).
    """
    if isinstance(obj, type) and issubclass(obj, Stub):
        return stub_class(f"{obj._qualname}.{name}")
    if isinstance(obj, Stub):
        return stub_class(f"{obj._qualname}.{name}")(obj)
    raise pickle.UnpicklingError(f"getattr({obj!r}, {name!r}) not allowed")


# --------------------------------------------------------------------------- unpickler

_ALLOWED = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("collections", "defaultdict"): collections.defaultdict,
    ("builtins", "dict"): dict,
    ("builtins", "list"): list,
    ("builtins", "set"): set,
    ("builtins", "frozenset"): frozenset,
    ("builtins", "tuple"): tuple,
    ("builtins", "int"): int,
    ("builtins", "float"): float,
    ("builtins", "bool"): bool,
    ("builtins", "str"): str,
    ("builtins", "bytes"): bytes,
    ("builtins", "slice"): slice,
    ("builtins", "getattr"): _safe_getattr,
    ("_codecs", "encode"): __import__("_codecs").encode,  # numpy bytes in protocol 2
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
    ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
    ("torch", "Size"): torch.Size,
    ("torch", "device"): torch.device,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
    ("torch._utils", "_rebuild_parameter_with_state"): torch._utils._rebuild_parameter_with_state,
}


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Python-2 names (`__builtin__.long`, ...) -> Python 3, as pickle does itself.
        module, name = _compat_pickle.NAME_MAPPING.get((module, name), (module, name))
        module = _compat_pickle.IMPORT_MAPPING.get(module, module)
        if (module, name) in _ALLOWED:
            return _ALLOWED[(module, name)]
        if module == "torch" and isinstance(getattr(torch, name, None), torch.dtype):
            return getattr(torch, name)  # e.g. torch.float32
        # (torch.*Storage is resolved by torch.load's own wrapper before reaching us.)
        if module.startswith("torch") and name.startswith("_rebuild"):
            # An unlisted tensor rebuild function (quantized, sparse, subclass, ...): a stub
            # would silently drop tensors, so fail loudly and extend _ALLOWED instead.
            raise pickle.UnpicklingError(f"tensor rebuild function {module}.{name} not allowed")
        return stub_class(f"{module}.{name}")


class _PickleModule:
    """What `torch.load(pickle_module=...)` needs."""

    Unpickler = Unpickler
    __name__ = "checkpoint"

    @staticmethod
    def load(f, **kw):
        return Unpickler(f, **kw).load()


def load(path, mmap=True):
    """Unpickle the checkpoint into a stub tree. Tensors are real, CPU, mmap-backed.

    Returns the root Stub. `root._stub_locations` lists the original storage devices.
    """
    locations = collections.Counter()

    def to_cpu(storage, location):
        locations[location] += 1
        return storage

    root = torch.load(path, map_location=to_cpu, pickle_module=_PickleModule, weights_only=False, mmap=mmap)
    root.__dict__["_stub_locations"] = dict(locations)
    return root


# --------------------------------------------------------------------------- identity

IDENTITY_FIELDS = ("uuid", "run_id", "size_bytes", "sha256")


def anemoi_metadata(path):
    """The zip member ``*/anemoi-metadata/anemoi.json`` as a dict (plain JSON; nothing is unpickled)."""
    with zipfile.ZipFile(path) as z:
        member = next((n for n in z.namelist() if n.endswith("anemoi-metadata/anemoi.json")), None)
        if member is None:
            raise ValueError(f"{path}: no anemoi-metadata/anemoi.json member (not an anemoi checkpoint?)")
        return json.loads(z.read(member))


def checkpoint_identity(path, sha256=True):
    """{path, uuid, run_id, size_bytes, sha256} of a checkpoint file (docs/contracts.md §1 "source").

    uuid and run_id come from the embedded anemoi.json; sha256 is over the file bytes (skip with
    sha256=False). 
    """
    path = Path(path)
    meta = anemoi_metadata(path)
    out = {"path": str(path.resolve()), "uuid": meta.get("uuid"), "run_id": meta.get("run_id"),
           "size_bytes": path.stat().st_size}
    if sha256:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 24):
                h.update(chunk)
        out["sha256"] = h.hexdigest()
    return out


# --------------------------------------------------------------------------- walking

_MODULE_INTERNALS = {"_parameters", "_buffers", "_modules"}
_BOILERPLATE = {"training", "_non_persistent_buffers_set", "_is_full_backward_hook", "_compiled_call_impl"}
_BACKREFS = {"_parent"}  # omegaconf / torch_geometric child -> parent links (cycles, no new data)


def is_module(obj):
    return isinstance(obj, Stub) and "_modules" in obj.__dict__ and "_parameters" in obj.__dict__


def _key(k):
    return "(" + ",".join(map(str, k)) + ")" if isinstance(k, tuple) else str(k)


def _join(prefix, k):
    return f"{prefix}.{_key(k)}" if prefix else _key(k)


def _stub_internals(d):
    """Recorded constructor args / non-dict state / items of a stub, when present."""
    for i, v in enumerate(d["_stub_args"]):
        yield f"__args__{i}", v
    for k, v in d["_stub_kwargs"].items():
        yield f"__kwargs__{k}", v
    if d["_stub_state"] is not None:
        yield "__state__", d["_stub_state"]
    yield from d["_stub_items"].items()
    yield from enumerate(d["_stub_list"])


def _children(obj):
    """(key, child, kind) for every reachable child.

    kind: param | buffer | buffer_np (non-persistent buffer) | module | attr.
    """
    if isinstance(obj, Stub):
        d = obj.__dict__
        mod = is_module(obj)
        if mod:
            for k, v in d["_parameters"].items():
                yield k, v, "param"
            np_set = d.get("_non_persistent_buffers_set", ())
            for k, v in d["_buffers"].items():
                yield k, v, "buffer_np" if k in np_set else "buffer"
            for k, v in d["_modules"].items():
                yield k, v, "module"
        for k, v in _stub_internals(d):
            yield k, v, "attr"
        for k, v in d.items():
            if k.startswith("_stub_") or k in _BACKREFS:
                continue
            if mod and (k in _MODULE_INTERNALS or k in _BOILERPLATE or "hook" in k):
                continue
            yield k, v, "attr"
    elif isinstance(obj, dict):
        yield from ((k, v, "attr") for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        yield from ((i, v, "attr") for i, v in enumerate(obj))


def walk(root, prefix=""):
    """Yield (path, obj, kind) depth-first, visiting each container/stub once.

    Tensors and arrays are yielded at every path they are reachable from (aliases).
    A stub reached again is yielded once more with kind "alias:<first path>" and not
    descended into (e.g. post_processors share the normalizer of pre_processors).
    """
    seen = {}

    def rec(obj, path, kind):
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            yield path, obj, kind
            return
        if not isinstance(obj, (Stub, dict, list, tuple)):
            return
        if id(obj) in seen:
            if isinstance(obj, Stub):
                yield path, obj, "alias:" + seen[id(obj)]
            return
        seen[id(obj)] = path
        yield path, obj, kind
        for k, v, kk in _children(obj):
            yield from rec(v, _join(path, k), kk)

    yield from rec(root, prefix, "module" if is_module(root) else "attr")


def aliases(root):
    """{path: first path} for every stub reachable at more than one path."""
    return {p: k[6:] for p, _, k in walk(root) if k.startswith("alias:")}


def tensor_entries(root):
    """{id: [primary_path, tensor, kind, alias_paths]} for every tensor and ndarray.

    The primary path is a param/buffer path if there is one, else the shortest path.
    """
    out = {}
    for path, obj, kind in walk(root):
        if not isinstance(obj, (torch.Tensor, np.ndarray)):
            continue
        if isinstance(obj, torch.nn.Parameter):
            kind = "param"
        e = out.setdefault(id(obj), [path, obj, kind, []])
        if e[0] == path:
            continue
        rank = lambda p, k: (k == "attr", p.count("."), len(p))  # noqa: E731
        if rank(path, kind) < rank(e[0], e[2]):
            e[3].append(e[0])
            e[0], e[2] = path, kind
        else:
            e[3].append(path)
    return out


def flat_tensors(root, numpy=False):
    """{dotted.path: tensor} of every tensor (and ndarray if numpy=True), deduplicated.

    An object reachable at several paths (shared module, aliased attribute) appears once;
    `tensor_entries` / `aliases` give the other paths.
    """
    return {
        p: t for p, t, _, _ in tensor_entries(root).values() if isinstance(t, torch.Tensor) or numpy
    }


def state_dict(root):
    """Parameters and persistent buffers: what `module.state_dict()` would have returned."""
    return {p: t for p, t, k, _ in tensor_entries(root).values() if k in ("param", "buffer")}


def summarize(v, maxlen=200):
    """Short repr for non-tensor attributes."""
    if isinstance(v, torch.Tensor):
        return f"tensor{tuple(v.shape)} {v.dtype}"
    if isinstance(v, np.ndarray):
        return f"ndarray{v.shape} {v.dtype}"
    if isinstance(v, type) and issubclass(v, Stub):
        return f"<class {v._qualname}>"
    if isinstance(v, Stub):
        return f"<{v._qualname}>"
    r = repr(v)
    return r if len(r) <= maxlen else r[: maxlen - 3] + "..."


def modules(root):
    """Yield (path, qualname, {attr: value}) for each module, non-tensor attributes only."""
    for path, obj, kind in walk(root):
        if is_module(obj) and not kind.startswith("alias:"):
            attrs = {
                k: v
                for k, v, kk in _children(obj)
                if kk == "attr" and not isinstance(v, (torch.Tensor, np.ndarray)) and not is_module(v)
            }
            yield path, obj._qualname, attrs


def to_plain(obj, _seen=None):
    """Best-effort conversion of a non-tensor stub subtree to plain Python for inspection.

    omegaconf nodes/containers -> their values, dict-like stubs -> dict, other stubs ->
    {"__class__": qualname, **attrs}. Tensors/arrays are returned as is; cycles -> "<cycle>".
    """
    _seen = set() if _seen is None else _seen
    if isinstance(obj, type) and issubclass(obj, Stub):
        return f"<class {obj._qualname}>"
    if isinstance(obj, (Stub, dict, list, tuple)):
        if id(obj) in _seen:
            return "<cycle>"
        _seen = _seen | {id(obj)}
    if isinstance(obj, Stub):
        d = obj.__dict__
        if "_val" in d:  # omegaconf ValueNode
            return to_plain(d["_val"], _seen)
        if "_content" in d:  # omegaconf DictConfig / ListConfig
            return to_plain(d["_content"], _seen)
        out = {"__class__": obj._qualname} if not d["_stub_items"] else {}
        for k, v, kind in _children(obj):
            if kind == "attr" and not is_module(v) and k != "_metadata":
                out[k] = to_plain(v, _seen)
        return out
    if isinstance(obj, dict):
        return {k: to_plain(v, _seen) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_plain(v, _seen) for v in obj)
    return obj


# --------------------------------------------------------------------------- CLI

if __name__ == "__main__":
    root = load(sys.argv[1])
    print("storage locations:", root._stub_locations)
    print("stubbed globals:", sorted(_stub_classes))
    print("\n# modules")
    for path, cls, attrs in modules(root):
        print(f"{path or '<root>'}: {cls}")
        for k, v in attrs.items():
            print(f"    {k} = {summarize(v)}")
    print("\n# shared stubs (path -> first path)")
    for p, first in aliases(root).items():
        print(f"{p} -> {first}")
    print("\n# tensors")
    n_param = 0
    for p, t, kind, al in tensor_entries(root).values():
        if kind == "param":
            n_param += t.numel()
        print(f"{p}\t{tuple(t.shape)}\t{t.dtype}\t{kind}" + (f"\taliases={al}" if al else ""))
    print("\nparameters:", n_param)
