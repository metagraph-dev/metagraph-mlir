############################
# Libraries used as plugins
############################

try:
    import mlir_graphblas as _

    has_mlir_graphblas = True
except ImportError:
    has_mlir_graphblas = False

from metagraph import PluginRegistry


def find_plugins():
    # Ensure we import all items we want registered
    from . import compiler, translators, types

    registry = PluginRegistry("metagraph_mlir")
    registry.register(compiler.MLIRCompiler())
    registry.register_from_modules(translators, types)
    return registry.plugins
