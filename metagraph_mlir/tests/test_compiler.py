import pytest
import metagraph as mg
from metagraph import abstract_algorithm, concrete_algorithm, PluginRegistry
from metagraph.core.resolver import Resolver
from metagraph.core.dask.resolver import DaskResolver
from metagraph.core.plugin import CompileError
from dask import delayed


import numpy as np
from metagraph.tests.util import default_plugin_resolver
from metagraph_mlir.compiler import (
    MLIRFunc,
    MLIRCompiler,
    SymbolTable,
    construct_call_wrapper_mlir,
    compile_wrapper,
)


def test_register(default_plugin_resolver):
    assert "mlir" in default_plugin_resolver.compilers


@pytest.mark.xfail
def test_compile_algorithm(res):
    # FIXME: replace this test
    a = np.arange(100, dtype=np.float32)
    ret = res.algos.testing.scale(a, 4.0)
    np.testing.assert_array_equal(ret, a * 4.0)


def test_construct_call_wrapper_mlir(ex1):
    # FIXME: replace this test
    tbl, (algo0, algo1) = ex1
    wrapper_func, constant_vals = construct_call_wrapper_mlir(
        wrapper_name="subgraph0",
        symbol_table=tbl,
        input_keys=["input0", "input1"],
        execute_keys=["algo0", "algo1"],
        output_key="algo1",
    )

    expected_mlir = """\
func @algo0(%arga: tensor<?xf32>, %argb: tensor<?xf64>, %argc: i32) -> tensor<?x?xf32> {
  %f0 = constant 0.0: f32
  %0 = splat %f0 : tensor<8x16xf32>
  %1 = tensor.cast %0 : tensor<8x16xf32> to tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

func @algo1(%arga: tensor<?x?xf32>, %argb: i32) -> tensor<?xf32> {
  %f0 = constant 0.0: f32
  %0 = splat %f0 : tensor<8xf32>
  %1 = tensor.cast %0 : tensor<8xf32> to tensor<?xf32>
  return %1 : tensor<?xf32>
}

func @subgraph0(%var0: tensor<?xf32>, %var1: tensor<?xf64>, %const0: i32, %const1: i32) -> tensor<?xf32> {
  %ret0 = call @algo0(%var0, %var1, %const0) : (tensor<?xf32>, tensor<?xf64>, i32) -> tensor<?x?xf32>
  %ret1 = call @algo1(%ret0, %const1) : (tensor<?x?xf32>, i32) -> tensor<?xf32>

  return %ret1 : tensor<?xf32>
}
"""
    assert wrapper_func.name == "subgraph0"
    assert wrapper_func.arg_types == ["tensor<?xf32>", "tensor<?xf64>", "i32", "i32"]
    assert wrapper_func.ret_type == "tensor<?xf32>"
    assert wrapper_func.mlir == expected_mlir
    assert constant_vals == [2, 5]


def test_compile_subgraph(dres):
    a = np.arange(100, dtype=np.float32)
    scale_func = dres.algos.testing.scale
    x = scale_func(a, 2.0)
    y = scale_func(x, 3.0)
    z = scale_func(y, 4.0)
    compiler = dres.compilers["mlir"]

    jit_func = compiler.compile_subgraph(z.__dask_graph__(), [], z.key)

    expected = a * 2.0 * 3.0 * 4.0
    ret = jit_func()
    np.testing.assert_array_equal(ret, expected)

    # confirm that kwargs are forbidden
    with pytest.raises(ValueError, match="kwargs"):
        jit_func(arg=1)


def test_compile_subgraph_with_input(dres):
    # FIXME: replace this test
    from metagraph.plugins.numpy.types import NumpyVectorType
    from metagraph.plugins.graphblas.types import GrblasVectorType

    a = np.arange(100, dtype=np.float32)
    # insert unnecesary translate to create "input" task for later
    a_translate = dres.translate(a, dst_type=GrblasVectorType)
    a_back = dres.translate(a_translate, dst_type=NumpyVectorType)

    scale_func = dres.algos.testing.scale
    x = scale_func(a_back, 2.0)
    y = scale_func(x, 3.0)
    z = scale_func(y, 4.0)

    subgraph = z.__dask_graph__().copy()
    assert len(subgraph) == 5
    # remove the translate tasks manually to make our subgraph
    del subgraph[a_translate.key]
    del subgraph[a_back.key]
    assert len(subgraph) == 3

    compiler = dres.compilers["mlir"]
    jit_func = compiler.compile_subgraph(subgraph, [a_back.key], z.key)

    expected = a * 2.0 * 3.0 * 4.0
    ret = jit_func(a)  # fused func expects one input, corresponding to a_back
    np.testing.assert_array_equal(ret, expected)


def test_compile_subgraph_kwargs_error(dres):
    a = np.arange(100, dtype=np.float32)
    x = dres.algos.testing.offset(a, offset=4.0)
    compiler = dres.compilers["mlir"]

    with pytest.raises(CompileError, match="offset"):
        jit_func = compiler.compile_subgraph(x.__dask_graph__(), [], x.key)


def test_compute(dres):
    a = np.arange(100, dtype=np.float32)
    scale_func = dres.algos.testing.scale
    x = scale_func(a, 2.0)
    y = scale_func(x, 3.0)
    z = scale_func(y, 4.0)

    expected = a * 2.0 * 3.0 * 4.0
    ret = z.compute()
    np.testing.assert_array_equal(ret, expected)


@pytest.fixture
def ex1():
    tbl = SymbolTable()
    tbl.register_var("input0", type="tensor<?xf32>")
    tbl.register_var("input1", type="tensor<?xf64>")

    algo0 = MLIRFunc(
        name="algo0",
        arg_types=["tensor<?xf32>", "tensor<?xf64>", "i32"],
        ret_type="tensor<?x?xf32>",
        mlir="""\
func @algo0(%arga: tensor<?xf32>, %argb: tensor<?xf64>, %argc: i32) -> tensor<?x?xf32> {
  %f0 = constant 0.0: f32
  %0 = splat %f0 : tensor<8x16xf32>
  %1 = tensor.cast %0 : tensor<8x16xf32> to tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
""",
    )

    algo1 = MLIRFunc(
        name="algo1",
        arg_types=["tensor<?x?xf32>", "i32"],
        ret_type="tensor<?xf32>",
        mlir="""\
func @algo1(%arga: tensor<?x?xf32>, %argb: i32) -> tensor<?xf32> {
  %f0 = constant 0.0: f32
  %0 = splat %f0 : tensor<8xf32>
  %1 = tensor.cast %0 : tensor<8xf32> to tensor<?xf32>
  return %1 : tensor<?xf32>
}
""",
    )

    tbl.register_func(
        "algo0",
        algo0,
        ["input0", "input1", 2],
        arg_types=["tensor<?xf32>", "tensor<?xf64>", "i32"],
        ret_type="tensor<?x?xf32>",
    )
    tbl.register_func(
        "algo1",
        algo1,
        ["algo0", 5],
        arg_types=["tensor<?x?xf32>", "i32"],
        ret_type="tensor<?xf32>",
    )

    return tbl, (algo0, algo1)


@pytest.fixture
def res():
    from metagraph.plugins.core.types import Vector
    from metagraph.plugins.numpy.types import NumpyVectorType

    @abstract_algorithm("testing.add")
    def testing_add(a: Vector, b: Vector) -> Vector:  # pragma: no cover
        pass

    @concrete_algorithm("testing.add", compiler="mlir")
    def compiled_add(
        a: NumpyVectorType, b: NumpyVectorType
    ) -> NumpyVectorType:  # pragma: no cover
        return MLIRFunc(
            name="testing_add",
            arg_types=["tensor<?xf32>", "tensor<?xf32>"],
            ret_type="tensor<?xf32>",
            mlir=b"""\
#trait_testing_add = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // A
    affine_map<(i) -> (i)>,  // B
    affine_map<(i) -> (i)>   // X (out)
  ],
  iterator_types = ["parallel"],
  doc = "X(i) = A(i) OP B(i)"
}

func private @testing_add(%arga: tensor<?xf32>, %argb: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic #trait_testing_add
     ins(%arga, %argb: tensor<?xf32>, tensor<?xf32>)
    outs(%arga: tensor<?xf32>) {
      ^bb(%a: f32, %b: f32, %s: f32):
        %0 = addf %a, %b  : f32
        linalg.yield %0 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
""",
        )

    @abstract_algorithm("testing.scale")
    def testing_scale(a: Vector, scale: float) -> Vector:  # pragma: no cover
        pass

    @concrete_algorithm("testing.scale", compiler="mlir")
    def compiled_scale(
        a: NumpyVectorType, scale: float
    ) -> NumpyVectorType:  # pragma: no cover
        return MLIRFunc(
            name="testing_scale",
            arg_types=["tensor<?xf32>", "f32"],
            ret_type="tensor<?xf32>",
            mlir="""\
#trait_testing_scale = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // A
    affine_map<(i) -> (i)>   // X (out)
  ],
  iterator_types = ["parallel"],
  doc = "X(i) = A(i) OP Scalar"
}

func private @testing_scale(%input: tensor<?xf32>, %scale: f32) -> tensor<?xf32> {
  %0 = linalg.generic #trait_testing_scale
     ins(%input: tensor<?xf32>)
     outs(%input: tensor<?xf32>) {
      ^bb(%a: f32, %s: f32):
        %0 = mulf %a, %scale  : f32
        linalg.yield %0 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
""",
        )

    @abstract_algorithm("testing.offset")
    def testing_offset(a: Vector, *, offset: float) -> Vector:  # pragma: no cover
        pass

    @concrete_algorithm("testing.offset", compiler="mlir")
    def compiled_offset(
        a: NumpyVectorType, *, offset: float
    ) -> NumpyVectorType:  # pragma: no cover
        return MLIRFunc(
            name="testing_offset",
            arg_types=["tensor<?xf32>", "f32"],
            ret_type="tensor<?xf32>",
            mlir=b"""\
#trait_testing_offset = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // A
    affine_map<(i) -> (i)>   // X (out)
  ],
  iterator_types = ["parallel"],
  doc = "X(i) = A(i) OP Scalar"
}

func private @testing_offset(%input: tensor<?xf32>, %offset: f32) -> tensor<?xf32> {
  %0 = linalg.generic #trait_testing_offset
     ins(%input: tensor<?xf32>)
     outs(%input: tensor<?xf32>) {
      ^bb(%a: f32, %s: f32):
        %0 = addf %a, %offset  : f32
        linalg.yield %0 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
""",
        )

    registry = PluginRegistry("test_subgraphs_plugin")
    registry.register(testing_add)
    registry.register(compiled_add)
    registry.register(testing_scale)
    registry.register(compiled_scale)
    registry.register(testing_offset)
    registry.register(compiled_offset)

    resolver = Resolver()
    # MLIRCompiler will be picked up from environment
    resolver.load_plugins_from_environment()
    resolver.register(registry.plugins)

    return resolver


@pytest.fixture
def dres(res):
    return DaskResolver(res)
