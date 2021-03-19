############################
# Libraries used as plugins
############################

try:
    import mlir_graphblas as _

    has_mlir_graphblas = True
except ImportError:
    has_mlir_graphblas = False

from metagraph import PluginRegistry

# Use this as the entry_point object
registry = PluginRegistry("metagraph_mlir")


def find_plugins():
    # Ensure we import all items we want registered
    from . import compiler, translators, types

    registry.register_from_modules(compiler, translators, types)
    return registry.plugins
