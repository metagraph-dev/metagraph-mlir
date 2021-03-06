"""
MLIR compiler plugin:

"""
from metagraph_numba.compiler import SymbolTable
from metagraph.core.plugin import Compiler, CompileError
from typing import Dict, List, Tuple, Callable, Hashable, Any
from dask.core import toposort, ishashable
from dataclasses import dataclass, field


@dataclass
class MLIRFunc:
    # name of function in mlir to call
    name: str

    # type signature of arguments to entrypoint (MLIR type syntax)
    # FIXME: Should we parse this from the MLIR directly?
    arg_types: List[str]

    # type signature of return type from entrypoint (MLIR type syntax)
    ret_type: str

    # MLIR text
    mlir: bytes


def mlir_func_decl_arg(name, mlir_type):
    return f"%{name}: {mlir_type}"


def construct_call_wrapper_mlir(
    wrapper_name: str,
    symbol_table: SymbolTable,
    input_keys: List[Hashable],
    execute_keys: List[Hashable],
    output_key: Hashable,
) -> Tuple[MLIRFunc, List]:
    constant_sym = []
    constant_vals = []
    for sym, val in symbol_table.const_sym_to_value.items():
        constant_sym.append(sym)
        constant_vals.append(val)

    func_text = ""

    # wrapper arg types
    decl_args = []
    arg_types = []
    for symbol_name in [
        symbol_table.var_key_to_sym[ikey] for ikey in input_keys
    ] + constant_sym:
        symbol_type = symbol_table.sym_to_type[symbol_name]
        decl_args.append(mlir_func_decl_arg(symbol_name, symbol_type))
        arg_types.append(symbol_type)

    # wrapper return type
    out_func_sym = symbol_table.func_key_to_sym[output_key]
    out_ret_sym = symbol_table.func_sym_to_ret_sym[out_func_sym]
    decl_ret_type = symbol_table.sym_to_type[out_ret_sym]

    # call signature
    func_text += (
        f"func @{wrapper_name}(" + ", ".join(decl_args) + f") -> {decl_ret_type} " "{\n"
    )

    # body
    for ekey in execute_keys:
        func_sym = symbol_table.func_key_to_sym[ekey]
        ret_sym = symbol_table.func_sym_to_ret_sym[func_sym]
        args_sym = symbol_table.func_sym_to_args_sym[func_sym]
        func_text += (
            f"  %{ret_sym} = call @{func_sym}("
            + ", ".join(["%" + s for s in args_sym])
            + ") : "
        )
        func_text += (
            "(" + ", ".join([symbol_table.sym_to_type[s] for s in args_sym]) + ") -> "
        )
        func_text += symbol_table.sym_to_type[ret_sym] + "\n"
    func_text += "\n"

    # return value
    func_text += f"  return %{out_ret_sym} : {decl_ret_type}" "\n}\n"

    mlir_func = MLIRFunc(
        name=wrapper_name, arg_types=arg_types, ret_type=decl_ret_type, mlir=func_text
    )
    return mlir_func, constant_vals


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
