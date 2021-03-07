"""
MLIR compiler plugin:

"""
from metagraph_numba.compiler import SymbolTable
from metagraph.core.plugin import Compiler, CompileError
from typing import Dict, List, Tuple, Callable, Hashable, Any
from dask.core import toposort, ishashable
from dataclasses import dataclass, field
from .adapters import AdapterRegistry, get_default_adapters


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
    func_definitions = {}

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
        func_impl = symbol_table.func_sym_to_func[func_sym]

        # save this function to be emitted with wrapper
        # key by function name to remove duplication of repeat functions
        func_definitions[func_impl.name] = func_impl.mlir

        # compute function type signature
        func_sig = (
            "("
            + ", ".join([symbol_table.sym_to_type[s] for s in args_sym])
            + ") -> "
            + symbol_table.sym_to_type[ret_sym]
        )

        # Use actual function name, rather than symbol assigned by symbol table
        func_text += (
            f"  %{ret_sym} = call @{func_impl.name}("
            + ", ".join(["%" + s for s in args_sym])
            + f") : {func_sig}"
            "\n"
        )

    func_text += "\n"

    # return value
    func_text += f"  return %{out_ret_sym} : {decl_ret_type}" "\n}\n"

    # prepend function definitons
    func_def_text = "\n".join(d for d in func_definitions.values())
    func_text = func_def_text + "\n" + func_text

    mlir_func = MLIRFunc(
        name=wrapper_name, arg_types=arg_types, ret_type=decl_ret_type, mlir=func_text
    )
    return mlir_func, constant_vals


class MLIRWrapper:
    def __init__(
        self,
        adapter_registry: AdapterRegistry,
        ctypes_func,
        mlir_func: MLIRFunc,
        const_values: List[Any],
    ):
        """Python callable that wraps an MLIR function

        Assumes that arg list is ordered such that Python args come first,
        then compile time constants.  That is, __call__ will be called with
        len(mlir_func.arg_types) - len(const_values) arguments.
        """
        self.ctypes_func = ctypes_func
        self.mlir_func = mlir_func
        self.const_values = List[Any]
        self._iarg_const = len(mlir_func.arg_types) - len(const_values)

        self._arg_adapters = [
            adapter_registry.search_by_mlir_type(mlir_func.arg_types[iarg])
            for iarg in range(self._iarg_const)
        ]

        self._ret_adapter = adapter_registry.search_by_mlir_type(mlir_func.ret_type)

    def __call__(*args, **kwargs):
        if len(kwargs) > 0:
            raise ValueError("MLIRWrapper does not support kwargs")

        args = [a.unbox(v) for a, v in zip(self.arg_adapters, args[: self._iarg_const])]
        args += self._const_unboxed

        ret = self.ctypes_func(*args)
        return self._ret_adapter.box(ret)


def compile_wrapper(
    engine,
    adapter_registry: AdapterRegistry,
    mlir_func: MLIRFunc,
    constant_vals: List[Any],
) -> Callable:
    ctypes_func = engine.compile(mlir_func)

    wrapper = MLIRWrapper(adapter_registry, ctypes_func, mlir_func, constant_vals)

    return wrapper


class MLIRCompiler(Compiler):
    def __init__(self, name="mlir"):
        super().__init__(name=name)
        self._subgraph_count = 0
        self._adapter_registry = get_default_adapters()
        # FIXME: add engine!
        self._engine = None

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

            mlir_func = delayed_algo.algo.func(*args)
            tbl.register_func(key, mlir_func, args)

        # generate the wrapper
        subgraph_wrapper_name = "subgraph" + str(self._subgraph_count)
        self._subgraph_count += 1
        mlir_func, constant_vals = construct_call_wrapper_mlir(
            wrapper_name=subgraph_wrapper_name,
            symbol_table=tbl,
            input_keys=inputs,
            execute_keys=toposort_keys,
            output_key=output,
        )

        wrapper_func = compile_wrapper(
            self.engine, self.adapter_registry, mlir_func, constant_vals
        )

        return wrapper_func
