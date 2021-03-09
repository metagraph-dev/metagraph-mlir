from typing import Type, Optional, Any, List
import ctypes
import mlir
import numpy as np
import numba.types as nbtypes
import lark


class Adapter:
    def box(self, mlir_value) -> Any:
        """Convert a return value from MLIR (wrapped in ctypes) to a Python value."""
        raise NotImplementedError

    def unbox(self, python_value) -> List[Any]:
        """Convert a Python value to MLIR value(s) (wrapped in ctypes).

        The MLIR calling convention will sometimes unroll a single high level
        type into several sequential arguments.  (ex tensor -> 3 + 2 * dims arguments)
        For that reason, the unboxer returns a list of values."""
        raise NotImplementedError


class AdapterGenerator:
    def search_by_numba_type(self, numba_type: nbtypes.Type) -> Optional[Adapter]:
        """Return an adapter instance for this Numba type.
        
        Returns None if this generator cannot handle this type.
        """
        raise NotImplementedError

    def search_by_mlir_type(self, mlir_type: str) -> Optional[Adapter]:
        """Return an adapter instance for this MLIR type.

        Returns None if this generator cannot handle this type.
        """
        raise NotImplementedError


class AdapterRegistry:
    """
    FIXME: This function gives priority to the last registered adapter when
    there is ambiguous boxing!
    """

    def __init__(self):
        self._numba_typemap = {}
        self._mlir_typemap = {}
        self._generators = []

    def register_adapter(
        self, adapter: Adapter, mlir_type: str, numba_type: nbtypes.Type
    ):
        self._numba_typemap[numba_type] = adapter
        self._mlir_typemap[mlir_type] = adapter

    def register_generator(self, generator: AdapterGenerator):
        self._generators.append(generator)

    def search_by_python_value(self, python_value) -> Adapter:
        return self.search_by_numba_type(numba.typeof(python_value))

    def search_by_numba_type(self, numba_type: nbtypes.Type) -> Adapter:
        adapter = self._numba_typemap.get(numba_type, None)
        if adapter is not None:
            return adapter

        for generator in self._generators:
            adapter = generator.search_by_numba_type(numba_type)
            if adapter is not None:
                return adapter

        raise TypeError(f"Unable to find adapter for Numba type {numba_type}")

    def search_by_mlir_type(self, mlir_type: str) -> Adapter:
        adapter = self._mlir_typemap.get(mlir_type, None)
        if adapter is not None:
            return adapter

        for generator in self._generators:
            adapter = generator.search_by_mlir_type(mlir_type)
            if adapter is not None:
                return adapter

        raise TypeError(f"Unable to find adapter for MLIR type {mlir_type}")


###### Default adapter registry ######


class ScalarAdapter(Adapter):
    def __init__(self, ctypes_type, python_cast, dtype):
        self.ctypes_type = ctypes_type
        self.python_cast = python_cast
        self.dtype = dtype  # for use by array adapter

    def box(self, mlir_value) -> Any:
        # get the Python value from the ctypes object, then coerce it to
        # specified Python type
        return self.python_cast(mlir_value.value)

    def unbox(self, python_value) -> List[Any]:
        return [python_value]
        # Scalars always unbox to single value
        # return [self.ctypes_type(python_value)]


class ArrayAdapter(Adapter):
    def __init__(self, dtype, ndims, ctypes_element_type, ctypes_index_type):
        self._dtype = dtype
        self._ndims = ndims
        self._ctypes_element_type = ctypes_element_type
        self._ctypes_index_type = ctypes_index_type

        ctypes_fields = (
            [
                ("alloc", ctypes.POINTER(ctypes_element_type)),
                ("base", ctypes.POINTER(ctypes_element_type)),
                ("offset", ctypes_index_type),
            ]
            + [(f"size{d}", ctypes_index_type) for d in range(ndims)]
            + [(f"stride{d}", ctypes_index_type) for d in range(ndims)]
        )

        class ReturnType(ctypes.Structure):
            _fields_ = ctypes_fields

        self.ctypes_struct_type = ReturnType

    def box(self, mlir_value):
        return mlir_value
        # FIXME: redundant with MlirJitEngine
        # FIXME: Does not handle strides or memory lifetime yet!  Need cython!
        shape = []
        nelements = 1
        for d in range(self._ndims):
            size = getattr(mlir_value, f"size{d}")
            shape.append(size)
            nelements *= size
        ret = np.frombuffer(
            (self._ctypes_element_type * nelements).from_address(
                ctypes.addressof(mlir_value.base.contents)
            ),
            dtype=self._dtype,
        ).reshape(tuple(shape))
        return ret

    def unbox(self, python_value):
        # FIXME: MlirJitEngine is doing the unpacking already
        return [python_value]
        # assume that function signature is using np.ctypeslib.ndpointer
        itemsize = python_value.itemsize
        args = (
            [python_value, python_value, 0,]
            + list(python_value.shape)
            + [python_value.strides[d] // itemsize for d in range(self._ndims)]
        )

        return args


class MLIRTypeParser:
    def __init__(self):
        # get the Lark parser from pyMLIR
        self._mlir_parser = mlir.Parser()
        self._lark_parser = self._mlir_parser.parser
        self._mlir_transformer = self._mlir_parser.transformer

    def parse(self, mlir_type):
        try:
            parsed_type = self._lark_parser.parse(mlir_type, "type")
            node = self._mlir_transformer.transform(parsed_type)
            return node
        except lark.UnexpectedInput:
            return None


class ArrayGenerator(AdapterGenerator):
    def __init__(self, scalar_registry: AdapterRegistry):
        """Generate adapters for NumPy arrays.

        Scalar registry is used to map between MLIR and Numba element types
        """
        self._scalar_registry = scalar_registry
        self._parser = MLIRTypeParser()

    def search_by_numba_type(self, numba_type: nbtypes.Type) -> Adapter:
        if isinstance(numba_type, nbtypes.Array):
            element_type = numba_type.dtype  # not NumPy dtype
            scalar_adapter = self._scalar_registry.search_by_numba_type(element_type)
            if scalar_adapter is None:
                return None

            adapter = ArrayAdapter(
                dtype=scalar_adapter.dtype,
                ndims=numba_type.ndim,
                ctypes_element_type=scalar_adapter.ctypes_type,
                ctypes_index_type=ctypes.c_int64,
            )

            return adapter
        else:
            return None

    def search_by_mlir_type(self, mlir_type: str) -> Optional[Adapter]:
        """Return an adapter instance for this MLIR type.

        Returns None if this generator cannot handle this type.
        """
        parsed_type = self._parser.parse(mlir_type)
        if isinstance(parsed_type, mlir.astnodes.RankedTensorType):
            ndims = len(parsed_type.dimensions)
            element_type = parsed_type.element_type.type.name

            scalar_adapter = self._scalar_registry.search_by_mlir_type(element_type)
            if scalar_adapter is None:
                return None

            adapter = ArrayAdapter(
                dtype=scalar_adapter.dtype,
                ndims=ndims,
                ctypes_element_type=scalar_adapter.ctypes_type,
                ctypes_index_type=ctypes.c_int64,
            )
            return adapter
        else:
            return None


def get_default_adapters() -> AdapterRegistry:
    reg = AdapterRegistry()
    # use scalar_reg for arraygenerator
    scalar_reg = AdapterRegistry()

    # Scalars
    for mtype, ctype, nbtype, dtype in [
        # Integer
        ("i8", ctypes.c_int8, nbtypes.int8, np.int8),
        ("i16", ctypes.c_int16, nbtypes.int16, np.int16),
        ("i32", ctypes.c_int32, nbtypes.int32, np.int32),
        ("i64", ctypes.c_int64, nbtypes.int64, np.int64),
        # Float
        ("f32", ctypes.c_float, nbtypes.float32, np.float32),
        ("f64", ctypes.c_double, nbtypes.float64, np.float64),
    ]:
        adapter = ScalarAdapter(ctypes_type=ctype, python_cast=dtype, dtype=dtype)
        reg.register_adapter(adapter, mlir_type=mtype, numba_type=nbtype)
        scalar_reg.register_adapter(adapter, mlir_type=mtype, numba_type=nbtype)

    # NumPy arrays
    reg.register_generator(ArrayGenerator(scalar_registry=scalar_reg))

    return reg
