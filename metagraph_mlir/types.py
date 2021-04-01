from typing import Set, Dict, Any
from metagraph import dtypes
from metagraph.plugins.core.types import Graph
from metagraph.plugins.core.wrappers import GraphWrapper
from .registry import has_mlir_graphblas
import numpy as np

if has_mlir_graphblas:
    from mlir_graphblas.sparse_utils import MLIRSparseTensor

    class MLIRGraphBLASGraph(GraphWrapper, abstract=Graph):
        def __init__(
            self,
            data,
            is_weighted: bool,
            node_list=None,
            node_vals=None,
            *,
            aprops=None,
        ):
            # TODO need is_weighted (for edges) as MLIR's sparse
            # tensors don't support integer or boolean weights.
            super().__init__(aprops=aprops)
            self._assert_instance(data, MLIRSparseTensor)
            self._assert_instance(is_weighted, bool)
            self._assert(data.ndim == 2, f"Sparse adjacency matrix must have rank 2.")
            nrows, ncols = data.shape
            self._assert(nrows == ncols, "Sparse adjacency matrix must be square")
            if node_list is None:
                node_list = np.arange(nrows)
            else:
                self._assert_instance(node_list, (np.ndarray, list, tuple))
                if not isinstance(node_list, np.ndarray):
                    node_list = np.array(node_list)
            if node_vals is not None:
                self._assert_instance(node_vals, (np.ndarray, list, tuple))
                if not isinstance(node_vals, np.ndarray):
                    node_vals = np.array(node_vals)
                self._assert(
                    nrows == len(node_vals),
                    f"node vals size ({len(node_vals)}) and data matrix size ({nrows}) don't match",
                )
            self.value = data
            self.is_weighted = is_weighted
            # TODO when sparsity in all dimensions is supported in the ScipyGraph <-> MLIRGraphBLASGraph
            # translators, will we still need node_list? Or just node_vals?
            self.node_list: np.ndarray = node_list
            self.node_vals: np.ndarray = node_vals

        @property
        def __mlir_sparse__(self):
            return self.value

        class TypeMixin:
            @classmethod
            def _compute_abstract_properties(
                cls, obj, props: Set[str], known_props: Dict[str, Any]
            ) -> Dict[str, Any]:
                ret = known_props.copy()

                # fast properties
                for prop in {
                    "node_type",
                    "node_dtype",
                    "edge_type",
                    "edge_dtype",
                } - ret.keys():
                    if prop == "node_type":
                        ret[prop] = "set" if obj.node_vals is None else "map"
                    elif prop == "node_dtype":
                        if obj.node_vals is None:
                            ret[prop] = None
                        else:
                            ret[prop] = dtypes.dtypes_simplified[obj.node_vals.dtype]
                    elif prop == "edge_type":
                        ret[prop] = "map" if obj.is_weighted else "set"
                    elif prop == "edge_dtype":
                        if obj.is_weighted:
                            ret[prop] = dtypes.dtypes_simplified[obj.value.values.dtype]
                        else:
                            ret[prop] = None

                # slow properties, only compute if asked
                for prop in props - ret.keys():
                    if prop == "is_directed":
                        indices = set(obj.indices)
                        ret[prop] = all(index[::-1] in indices for index in indices)
                    elif prop == "has_negative_weights":
                        if ret["dtype"] in {"bool", "str"}:
                            neg_weights = None
                        else:
                            min_val = obj.value.values.min()
                            if min_val < 0:
                                neg_weights = True
                            else:
                                neg_weights = False
                        ret[prop] = neg_weights

                return ret

            @classmethod
            def assert_equal(
                cls,
                obj1,
                obj2,
                aprops1,
                aprops2,
                cprops1,
                cprops2,
                *,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                m1, m2 = obj1.value, obj2.value
                assert m1.ndim == m2.ndim, f"size mismatch: {m1.ndim} != {m2.ndim}"
                m1_nnz, m2_nnz = len(m1.values), len(m2.values)
                assert m1_nnz == m2_nnz, f"num edges mismatch: {m1_nnz} != {m2_nnz}"
                assert (
                    m1.index_dtype == m2.index_dtype
                ), f"index type mismatch {m1.index_dtype} == {m2.index_dtype}"
                assert (
                    m1.value_dtype == m2.value_dtype
                ), f"value type mismatch {m1.index_dtype} == {m2.index_dtype}"
                assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
                # TODO is sorting in C/C++/Cython faster than using a set?
                m1_indices, m2_indices = set(m1.indices), set(m2.indices)
                assert (
                    m1_indices == m2_indices
                ), f"edge mismatch {m1_indices} != {m2_indices}"
                if aprops1["edge_type"] == "map":
                    sort1 = np.argsort(m1.indices)
                    sort2 = np.argsort(m2.indices)
                    vals1 = m1.values[sort1]
                    vals2 = m2.values[sort2]
                    if issubclass(vals1.dtype.type, np.floating):
                        assert np.isclose(
                            vals1, vals2, rtol=rel_tol, atol=abs_tol
                        ).all()
                    else:
                        assert (vals1 == vals2).all()
                if aprops1["node_type"] == "map":
                    sort1 = np.argsort(obj1.node_list)
                    sort2 = np.argsort(obj2.node_list)
                    vals1 = obj1.node_vals[sort1]
                    vals2 = obj2.node_vals[sort2]
                    if issubclass(vals1.dtype.type, np.floating):
                        assert np.isclose(
                            vals1, vals2, rtol=rel_tol, atol=abs_tol
                        ).all()
                    else:
                        assert (vals1 == vals2).all()
