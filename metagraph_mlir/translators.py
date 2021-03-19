from metagraph import translator
from metagraph.plugins import has_scipy
from .registry import has_mlir_graphblas
import numpy as np

if has_mlir_graphblas:
    from mlir_graphblas.sparse_utils import MLIRSparseTensor
    from .types import MLIRGraphBLASGraph

if has_mlir_graphblas and has_scipy:

    import scipy.sparse as ss
    from metagraph.plugins.scipy.types import ScipyGraph

    @translator
    def mlir_from_scipy(x: ScipyGraph, **props) -> MLIRGraphBLASGraph:
        aprops = ScipyGraph.Type.compute_abstract_properties(
            x, {"node_type", "edge_type", "node_dtype", "edge_dtype", "is_directed"}
        )
        if aprops["edge_type"] == "map" and x.value.dtype not in (
            np.float32,
            np.float64,
        ):
            raise NotImplementedError(
                f"{MLIRGraphBLASGraph} instances with {aprops['edge_dtype']} edge weights not supported."
            )

        matrix = x.value.tocoo()
        indices = np.column_stack([matrix.row, matrix.col]).astype(np.uint64)
        values = matrix.data
        sizes = np.array(matrix.shape, dtype=np.uint64)
        sparsity = np.array(
            [False, True], dtype=np.bool8
        )  # TODO Make this sparse in all dimensions; [False, True] is equivalent to CSR
        mlir_sparse_tensor = MLIRSparseTensor(indices, values, sizes, sparsity)

        is_weighted = aprops["edge_type"] == "map"

        node_list = np.copy(x.node_list)
        if aprops["node_type"] == "map":
            node_vals = np.copy(x.node_vals)
        else:
            node_vals = None

        return MLIRGraphBLASGraph(
            mlir_sparse_tensor, is_weighted, node_list, node_vals, aprops=aprops
        )

    @translator
    def mlir_to_scipy(x: MLIRGraphBLASGraph, **props) -> ScipyGraph:
        aprops = MLIRGraphBLASGraph.Type.compute_abstract_properties(
            x, {"node_type", "edge_type", "node_dtype", "edge_dtype", "is_directed"}
        )

        d0_indices, d1_indices = x.value.indices
        d0_pointers, d1_pointers = x.value.pointers
        values = x.value.values
        if not (
            len(d0_indices) == 0
            and len(d0_pointers) == 0
            and len(d1_indices) != 0
            and len(d1_pointers) != 0
        ):
            # TODO Handle all 4 cases:
            #     Sparse Rows + Dense Columns (i.e. CSR)
            #     Dense Rows + Sparse Columns (i.e. CSC)
            #     Sparse Rows + Sparse Columns (i.e. hypersparse)
            #     Dense Rows + Dense Columns (i.e. dense)
            raise NotImplementedError(
                f"Translations for {MLIRGraphBLASGraph} instances not dense in the first "
                "dimension and sparse in the second dimension, i.e. CSR format, not yet supported."
            )

        m = ss.csr_matrix(
            (values, d1_indices, d1_pointers), dtype=values.dtype, shape=x.value.shape,
        )

        node_list = np.copy(x.node_list)
        if aprops["node_type"] == "map":
            node_vals = np.copy(x.node_vals)
        else:
            node_vals = None

        return ScipyGraph(m, node_list, node_vals, aprops=aprops)
