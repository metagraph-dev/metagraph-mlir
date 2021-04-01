
metagraph-mlir Documentation
============================

metagraph-mlir is a plugin for `Metagraph`_ that enables algorithms to be
just-in-time (JIT) compiled with `MLIR`_.  To learn more about compiler
plugins in Metagraph, see the `Compiler Plugins`_ section of the Metagraph
Plugin Author Guide. 

metagraph-mlir is licensed under the `Apache 2.0 license`_ and the source
code can be found on `Github`_.


.. _MLIR: https://mlir.llvm.org/
.. _Metagraph: https://metagraph.readthedocs.org
.. _metagraph-mlir: https://metagraph-mlir.readthedocs.org
.. _Compiler Plugins: https://metagraph.readthedocs.org/en/plugin_author_guide/compiler_plugins.html
.. _Apache 2.0 license: https://www.apache.org/licenses/LICENSE-2.0
.. _Github: https://github.com/metagraph-dev/metagraph-mlir


Installation
------------

metagraph-mlir is currently only distributed via conda.  To install::

    conda install -c metagraph -c conda-forge metagraph-mlir

Note that metagraph-mlir requires a development version of MLIR from the
unreleased LLVM 12.0.  This version is automatically installed from the
metagraph channel by conda.

Implementing Algorithms with metagraph-mlir
-------------------------------------------

The metagraph-mlir compiler requires that the concrete algorithm (learn more
about Metagraph algorithms `here <https://metagraph.readthedocs.io/en/latest/user_guide/algorithms.html>`_) function return a
``metagraph_mlir.compiler.MLIRFunc`` object with the implementation of the
algorithm in MLIR.  For example, suppose an abstract algorithm has already
been defined to add two vectors:

.. code-block:: python

   @abstract_algorithm("example.add")
   def example_add(a: Vector, b: Vector) -> Vector:
       pass

A concrete implementation of this algorithm in MLIR be written this way:

.. code-block:: python

    @concrete_algorithm("example.add", compiler="mlir")
    def compiled_add(
        a: NumpyVectorType, b: NumpyVectorType
    ) -> NumpyVectorType:
        return MLIRFunc(
            name="example_add",
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

        func private @example_add(%arga: tensor<?xf32>, %argb: tensor<?xf32>) -> tensor<?xf32> {
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

This `MLIRFunc` object returned by ``compile_add`` in this example contains a
number of special attributes describing the function, including:

  * ``name``: Name of the main entry point function.
  * ``arg_types``: A list of strings containing the MLIR types for each function argument.
  * ``ret_type``: A string containing the MLIR type of return value
  * ``mlir``: A bytes object (not string) containing the text of the MLIR code.

To minimize the size of the JIT compiled modules (and to simplify inspection
of the MLIR after inlining into the generated wrapper function), all MLIR
functions in the body should be defined as ``private``.

The MLIR functions are JIT compiled using the `JIT engine in mlir-graphblas`_.
See that documentation for the examples of how to write functions with scalars,
dense tensors, and sparse tensors.

.. _JIT engine in mlir-graphblas: https://mlir-graphblas.readthedocs.io/en/latest/tools/engine.html


Translating between ScipyGraph and MLIRGraphBLASGraph
-----------------------------------------------------

metagraph-mlir currently supports translations between graphs of type ``ScipyGraph`` (provided as a core type in `Metagraph <https://metagraph.readthedocs.io>`_) and ``MLIRGraphBLASGraph`` (provided by metagraph-mlir).

``MLIRGraphBLASGraph`` is an adjacency matrix representation of a graph implemented via MLIR's current sparse tensor support. `This tutorial <https://mlir-graphblas.readthedocs.io/en/latest/tools/engine/spmv.html>`_ provides an overview of how MLIR currently supports sparse tensors. It also shows examples of how to generate an instance of an MLIR sparse tensor. 

``MLIRGraphBLASGraph`` alows us to wrap an MLIR sparse tensor as a graph, e.g. 

.. code-block:: python

    import numpy as np
    import mlir_graphblas
    from mlir_graphblas.sparse_utils import MLIRSparseTensor
    import metagraph as mg
    
    # The sparse adjacency matrix below looks like this (where the underscores represent zeros):
    #
    # [[ 1.2, ___, ___, ___ ], 
    #  [ ___, ___, ___, 3.4 ], 
    #  [ ___, ___, 5.6, ___ ], 
    #  [ ___, ___, ___, ___ ]]
    #
    
    indices = np.array([
        [0, 0],
        [1, 3],
        [2, 2],
    ], dtype=np.uint64)
    values = np.array([1.2, 3.4, 5.6], dtype=np.float32)
    sizes = np.array([4, 4], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)
    
    sparse_tensor = mlir_graphblas.sparse_utils.MLIRSparseTensor(indices, values, sizes, sparsity)
    
    has_weighted_edges = True
    graph = mg.wrappers.Graph.MLIRGraphBLASGraph(sparse_tensor, has_weighted_edges, aprops={
        "node_type": "set",
        "node_dtype": None,
        "edge_type": "map",
        "edge_dtype": "float",
        "is_directed": True,
    })

We can translate this graph into a ``ScipyGraph`` like so:

.. code-block:: python

    scipy_graph = mg.translate(g, mg.wrappers.Graph.ScipyGraph)

This will allow us verify the following:

.. code-block:: python

    assert np.isclose(
            scipy_graph.value.toarray(),
            np.array([
                [1.2, 0. , 0. , 0. ],
                [0. , 0. , 0. , 3.4],
                [0. , 0. , 5.6, 0. ],
                [0. , 0. , 0. , 0. ]
            ])).all()
    
We can also translate back easily:    

.. code-block:: python

    graph_round_trip = mg.translate(scipy_graph, mg.wrappers.Graph.MLIRGraphBLASGraph)

There are some limitations:

  * Currently, we can only translate from ``MLIRGraphBLASGraph`` to ``ScipyGraph`` if the ``MLIRGraphBLASGraph`` instance's underlying sparse tensor is dense in the first dimension and sparse in the second dimension, i.e. if it is in `CSR format <https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_%28CSR,_CRS_or_Yale_format%29>`_. 
  * ``MLIRGraphBLASGraph`` currently only supports 32-bit and 64-bit floating point edge weights. Thus, when translating to ``MLIRGraphBLASGraph``, exceptions will be raised if the source type instance has boolean, string, or integer edge weights. 
