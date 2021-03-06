"""
MLIR compiler plugin:

"""
from metagraph_numba.compiler import SymbolTable
from metagraph.core.plugin import Compiler, CompileError
from typing import Dict, List, Tuple, Callable, Hashable, Any
from dask.core import toposort, ishashable


def construct_call_wrapper_text(
    wrapper_name: str,
    symbol_table: SymbolTable,
    input_keys: List[Hashable],
    execute_keys: List[Hashable],
    output_key: Hashable,
) -> Tuple[str, Dict[str, Any]]:
    pass


def compile_wrapper(
    wrapper_name: str, wrapper_text: str, wrapper_globals: Dict[str, Any]
) -> Callable:
    pass


class MLIRCompiler(Compiler):
    def __init__(self, name="mlir"):
        super().__init__(name=name)
        self._subgraph_count = 0

    def compile_algorithm(self, algo, literals):
        """Wrap a single function for JIT compilation and execution.
        
        literals is not used for anything currently
        """
        raise CompileError("Not implemented")

    def compile_subgraph(
        self, subgraph: Dict, inputs: List[Hashable], output: Hashable
    ) -> Callable:
        """Fuse a subgraph of tasks into a single compiled function.

        It is assumed that the function will be called with values corresponding to
        `inputs` in the order they are given.
        """
        tbl = SymbolTable()

        # must populate the symbol table in toposort order
        toposort_keys = list(toposort(subgraph))

        # register the inputs as variables
        for key in inputs:
            tbl.register_var(key)

        # register each function in the subgraph
        for key in toposort_keys:
            task = subgraph[key]
            # all metagraph tasks are in (func, args, kwargs) format
            delayed_algo, args, kwargs = task
            if isinstance(kwargs, tuple):
                # FIXME: why are dictionaries represented this way in the DAG?
                kwargs = kwargs[0](kwargs[1])
            if len(kwargs) != 0:
                raise CompileError(
                    "MLIRCompiler only supports functions with bound kwargs.\n"
                    f"When compiling:\n{delayed_algo.func_label}\nfound unbound kwargs:\n{kwargs}"
                )

            # FIXME: How should we pass args to this function?
            func_body = delayed_algo.algo.func(*args)
            tbl.register_func(key, func_body, args)

        # generate the wrapper
        subgraph_wrapper_name = "subgraph" + str(self._subgraph_count)
        self._subgraph_count += 1
        wrapper_text, wrapper_globals = construct_call_wrapper_text(
            wrapper_name=subgraph_wrapper_name,
            symbol_table=tbl,
            input_keys=inputs,
            execute_keys=toposort_keys,
            output_key=output,
        )

        wrapper_func = compile_wrapper(
            subgraph_wrapper_name, wrapper_text, wrapper_globals
        )
        pass
