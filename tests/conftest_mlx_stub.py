"""Early-load mlx stub for Linux CI (mlx is macOS-only).

This file is loaded as a pytest plugin via conftest.py or pytest's
``-p`` flag *before* any test-file-level imports happen, ensuring
``import mlx.core`` succeeds everywhere.
"""
import importlib.machinery
import sys
import types


def _install_mlx_stubs():
    """Install lightweight mlx module stubs into sys.modules."""
    try:
        import mlx.core  # noqa: F401
        return  # Real mlx available — nothing to do
    except (ImportError, OSError):
        pass

    # ---------------------------------------------------------------
    # Core fake array
    # ---------------------------------------------------------------
    class _FakeMXArray:
        def __init__(self, data):
            if isinstance(data, (list, tuple)):
                self._data = list(data)
            elif isinstance(data, (int, float)):
                self._data = [data]
            else:
                self._data = [data]

        @property
        def shape(self):
            d = self._data
            if isinstance(d, list) and d and isinstance(d[0], list):
                rows, cols = len(d), len(d[0])
                if cols and isinstance(d[0][0], list):
                    return (rows, cols, len(d[0][0]))
                return (rows, cols)
            return (len(d),) if isinstance(d, list) else ()

        @property
        def ndim(self):
            return len(self.shape)

        @property
        def size(self):
            r = 1
            for s in self.shape:
                r *= s
            return r

        def __getitem__(self, key):
            if isinstance(key, tuple):
                # Handle None (np.newaxis) indices by skipping them –
                # broadcast_to handles the actual shape expansion.
                real_keys = [k for k in key if k is not None]
                val = self._data
                for k in real_keys:
                    if isinstance(val, list):
                        val = val[k]
                    elif isinstance(val, _FakeMXArray):
                        val = val._data[k]
                if isinstance(val, list):
                    return _FakeMXArray(val)
                return _FakeMXArray(val)
            if key is None:
                return _FakeMXArray(self._data)
            return _FakeMXArray(self._data[key])

        def item(self):
            if isinstance(self._data, list):
                if len(self._data) == 1:
                    v = self._data[0]
                    return v.item() if isinstance(v, _FakeMXArray) else v
                raise ValueError("item() on multi-element array")
            return self._data

        def tolist(self):
            return [v.item() if isinstance(v, _FakeMXArray) else v for v in self._data]

        def __add__(self, other):
            if isinstance(other, _FakeMXArray):
                return _FakeMXArray([a + b for a, b in zip(self._data, other._data)])
            return _FakeMXArray([a + other for a in self._data])

        def __radd__(self, other):
            return self.__add__(other)

        def __len__(self):
            return len(self._data) if isinstance(self._data, list) else 1

        def __repr__(self):
            return f"_FakeMXArray({self._data})"

    # ---------------------------------------------------------------
    # Universal proxy: supports arbitrary attribute chains & calls
    # ---------------------------------------------------------------
    class _Proxy:
        """A no-op stand-in that supports any attribute access and calls."""

        def __init__(self, name="<proxy>"):
            object.__setattr__(self, "_name", name)

        def __repr__(self):
            return f"_Proxy({self._name})"

        def __getattr__(self, name):
            return _Proxy(f"{self._name}.{name}")

        def __call__(self, *a, **kw):
            return _Proxy(f"{self._name}()")

        def __bool__(self):
            return False

        def __iter__(self):
            return iter([])

        def __len__(self):
            return 0

    # ---------------------------------------------------------------
    # Module stubs
    # ---------------------------------------------------------------
    class _StubModule(types.ModuleType):
        """Module that returns a _Proxy for any missing attribute."""
        def __getattr__(self, name):
            return _Proxy(f"{self.__name__}.{name}")

    stub_names = [
        "mlx", "mlx.core", "mlx.core.metal", "mlx.core.random",
        "mlx.core.distributed",
        "mlx.nn", "mlx.nn.losses", "mlx.utils", "mlx.optimizers",
    ]
    stubs = {}
    for name in stub_names:
        m = _StubModule(name)
        # Give each stub a proper __spec__ so importlib.util.find_spec works
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        m.__path__ = []  # Mark as package so sub-imports work
        stubs[name] = m
        sys.modules[name] = m

    # Wire parent references
    stubs["mlx"].core = stubs["mlx.core"]
    stubs["mlx"].nn = stubs["mlx.nn"]
    stubs["mlx"].utils = stubs["mlx.utils"]
    stubs["mlx"].optimizers = stubs["mlx.optimizers"]
    stubs["mlx.core"].metal = stubs["mlx.core.metal"]
    stubs["mlx.core"].random = stubs["mlx.core.random"]
    stubs["mlx.core"].distributed = stubs["mlx.core.distributed"]
    stubs["mlx.nn"].losses = stubs["mlx.nn.losses"]

    # mlx.core attrs
    mx = stubs["mlx.core"]
    # Expose the class directly so isinstance(obj, mx.array) works.
    mx.array = _FakeMXArray

    def _zeros(shape, dtype=None):
        if isinstance(shape, tuple) and len(shape) == 2:
            return _FakeMXArray([[0] * shape[1] for _ in range(shape[0])])
        n = shape if isinstance(shape, int) else shape[0]
        return _FakeMXArray([0] * n)
    mx.zeros = _zeros

    def _broadcast_to(arr, shape):
        if not isinstance(arr, _FakeMXArray):
            return arr
        flat = arr._data
        while isinstance(flat, list) and flat and isinstance(flat[0], list):
            flat = [i for sub in flat for i in sub]
        if len(shape) == 3:
            dims, batch, seq = shape
            row = flat[:batch]
            result = []
            for _ in range(dims):
                dim_rows = []
                for val in row:
                    v = val.item() if isinstance(val, _FakeMXArray) else val
                    dim_rows.append([v] * seq)
                result.append(dim_rows)
            return _FakeMXArray(result)
        return arr
    mx.broadcast_to = _broadcast_to

    mx.int32 = "int32"
    mx.float32 = "float32"
    mx.float16 = "float16"
    mx.bfloat16 = "bfloat16"
    mx.uint32 = "uint32"
    mx.bool_ = "bool_"
    mx.inf = float("inf")
    mx.eval = lambda *a: None
    mx.compile = lambda fn=None, **kw: fn if fn else (lambda f: f)
    mx.stream = lambda *a, **kw: None
    mx.default_stream = lambda *a, **kw: None
    mx.new_stream = lambda *a, **kw: None
    mx.synchronize = lambda *a, **kw: None
    mx.reshape = lambda x, shape: x
    mx.concatenate = lambda arrays, axis=0: arrays[0] if arrays else _FakeMXArray([])
    mx.stack = lambda arrays, axis=0: _FakeMXArray([a._data for a in arrays])
    mx.expand_dims = lambda a, axis: a
    mx.squeeze = lambda a, axis=None: a
    mx.where = lambda cond, x, y: x
    mx.argmax = lambda x, axis=None: _FakeMXArray([0])
    mx.softmax = lambda x, axis=None: x
    mx.log = lambda x: x
    mx.exp = lambda x: x
    mx.abs = lambda x: x
    mx.sum = lambda x, axis=None: _FakeMXArray([0])
    mx.mean = lambda x, axis=None: _FakeMXArray([0])
    mx.max = lambda x, axis=None: _FakeMXArray([0])
    mx.min = lambda x, axis=None: _FakeMXArray([0])
    mx.arange = lambda *a, **kw: _FakeMXArray(list(range(*a)))
    mx.ones = lambda shape, dtype=None: _FakeMXArray([1] * (shape if isinstance(shape, int) else shape[0]))
    mx.full = lambda shape, val, dtype=None: _FakeMXArray([val] * (shape if isinstance(shape, int) else shape[0]))
    mx.eye = lambda n, dtype=None: _FakeMXArray([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    mx.stop_gradient = lambda x: x
    mx.metal = stubs["mlx.core.metal"]
    mx.random = stubs["mlx.core.random"]
    mx.distributed = stubs["mlx.core.distributed"]

    # mlx.core.metal attrs
    stubs["mlx.core.metal"].is_available = lambda: False
    stubs["mlx.core.metal"].device_info = lambda: {"memory_size": 0}
    stubs["mlx.core.metal"].get_active_memory = lambda: 0
    stubs["mlx.core.metal"].get_peak_memory = lambda: 0
    stubs["mlx.core.metal"].get_cache_memory = lambda: 0
    stubs["mlx.core.metal"].set_memory_limit = lambda *a, **kw: None
    stubs["mlx.core.metal"].set_cache_limit = lambda *a, **kw: None
    stubs["mlx.core.metal"].clear_cache = lambda: None

    # mlx.core.random
    stubs["mlx.core.random"].seed = lambda s: None

    # mlx.core.distributed
    stubs["mlx.core.distributed"].Group = type(
        "Group", (), {"__init__": lambda self, *a, **kw: None}
    )

    # mlx.nn.Module
    class _Module:
        def __init__(self):
            pass
        def parameters(self):
            return {}
        def update(self, params):
            pass
    stubs["mlx.nn"].Module = _Module
    stubs["mlx.nn"].Linear = type("Linear", (), {"__init__": lambda self, *a, **kw: None})
    stubs["mlx.nn"].Embedding = type("Embedding", (), {"__init__": lambda self, *a, **kw: None})
    stubs["mlx.nn"].RMSNorm = type("RMSNorm", (), {"__init__": lambda self, *a, **kw: None})
    stubs["mlx.nn"].quantize = lambda *a, **kw: None
    stubs["mlx.nn"].QuantizedLinear = type("QuantizedLinear", (), {"__init__": lambda self, *a, **kw: None})

    # mlx.utils
    stubs["mlx.utils"].tree_map = lambda fn, tree, *rest: tree
    stubs["mlx.utils"].tree_map_with_path = lambda fn, tree, *rest, **kw: tree
    stubs["mlx.utils"].tree_flatten = lambda tree, **kw: []
    stubs["mlx.utils"].tree_unflatten = lambda flat: {}
    stubs["mlx.utils"].tree_reduce = lambda fn, tree, **kw: 0

    # mlx.optimizers
    stubs["mlx.optimizers"].Adam = type("Adam", (), {"__init__": lambda self, *a, **kw: None})


# Install stubs immediately on import
_install_mlx_stubs()
