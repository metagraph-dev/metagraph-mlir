from metagraph import PluginRegistry


def find_plugins():
    # Ensure we import all items we want registered
    from . import compiler

    registry = PluginRegistry("metagraph_mlir")
    registry.register(compiler.MLIRCompiler())
    return registry.plugins
