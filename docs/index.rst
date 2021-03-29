
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
--------------------------------------------

The metagraph-mlir compiler requires that the decorated algorithm function
return a ``MLIRFunc`` object.  This object contains a number of special
attributes describing the algorithm, including:

  * ``name``: Name of the entrypoint function.
  * ``arg_types``: A list of strings containing the MLIR types for each function argument.
  * ``ret_type``: A string containing the MLIR type of return value
  * ``mlir``: A bytes object (not string) containing the text of the MLIR code.

To minimize the size of the JIT compiled modules (and to simplify inspection
of the MLIR after inlining into the generated wrapper function), all MLIR
functions in the body should be defined as ``private``.

As an example, suppose an abstract algorithm has already been defined to add
two vectors:

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

The MLIR functions are JIT compiled using the `JIT engine in mlir-graphblas`_.
See that documentation for the examples of how to write functions with scalars,
dense tensors, and sparse tensors.

.. _JIT engine in mlir-graphblas: https://mlir-graphblas.readthedocs.io/en/latest/tools/engine.html