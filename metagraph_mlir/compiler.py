"""
MLIR compiler plugin:

"""
from metagraph_numba.compiler import SymbolTable
from metagraph.core.plugin import Compiler, CompileError
from typing import Dict, List, Tuple, Callable, Hashable, Any
from dask.core import toposort, ishashable
from dataclasses import dataclass, field
import inspect


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

    # Full, compiled function wrapped for calling from Python
    callable: Callable = None


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
        mlir_func: MLIRFunc,
        const_values: List[Any],
    ):
        """Python callable that wraps an MLIR function

        Assumes that arg list is ordered such that Python args come first,
        then compile time constants.  That is, __call__ will be called with
        len(mlir_func.arg_types) - len(const_values) arguments.
        """
        self.mlir_func = mlir_func
        self.const_values = const_values

    def __call__(self, *args, **kwargs):
        if len(kwargs) > 0:
            raise ValueError("MLIRWrapper does not support kwargs")

        all_args = list(args) + self.const_values
        return self.mlir_func.callable(*all_args)


# FIXME: These should be configurable with donfig
STANDARD_OPT_PASSES = [
    "--linalg-bufferize",
    "--func-bufferize",
    "--finalizing-bufferize",
    "--convert-linalg-to-affine-loops",
    "--inline",
    "--affine-loop-fusion=fusion-maximal",
    "--memref-dataflow-opt",
    "--lower-affine",
    "--convert-scf-to-std",
    "--convert-std-to-llvm",
]


def compile_wrapper(
    engine,
    mlir_func: MLIRFunc,
    constant_vals: List[Any],
) -> Callable:

    engine.add(mlir_func.mlir, passes=STANDARD_OPT_PASSES)
    mlir_func.callable = engine[mlir_func.name]
    wrapper = MLIRWrapper(mlir_func, constant_vals)

    return wrapper


class MLIRCompiler(Compiler):
    def __init__(self, name="mlir"):
        super().__init__(name=name)
        self._subgraph_count = 0
        self._initialized = False
        self._engine = None

    def initialize_runtime(self):
        from mlir_graphblas import MlirJitEngine

        self._engine = MlirJitEngine()
        self._initialized = True

    def compile_algorithm(self, algo, literals):
        """Wrap a single function for JIT compilation and execution.

        literals is not used for anything currently
        """
        if not self._initialized:
            self.initialize_runtime()

        # We generate a wrapper for a single algo because algos
        # are currently marked as "private" functions, which will
        # be optimized away by MLIR unless something is calling them

        tbl = SymbolTable()

        sig = inspect.signature(algo.func)
        args = []
        input_keys = []
        for argname, arg in sig.parameters.items():
            tbl.register_var(argname)
            input_keys.append(argname)
            args.append(arg.annotation)

        # the arguments to the func are not used for anything yet. Could be
        # used in future to pass types and literals for codegen, like
        # numba.overload decorator
        mlir_func = algo.func(*args)
        tbl.register_func(
            mlir_func.name,
            mlir_func,
            input_keys,
            arg_types=mlir_func.arg_types,
            ret_type=mlir_func.ret_type,
        )

        # generate the wrapper
        subgraph_wrapper_name = "algo_call" + str(self._subgraph_count)
        self._subgraph_count += 1
        mlir_func, constant_vals = construct_call_wrapper_mlir(
            wrapper_name=subgraph_wrapper_name,
            symbol_table=tbl,
            input_keys=input_keys,
            execute_keys=[mlir_func.name],
            output_key=mlir_func.name,
        )

        wrapper_func = compile_wrapper(self._engine, mlir_func, constant_vals)
        return wrapper_func

    def compile_subgraph(
        self, subgraph: Dict, inputs: List[Hashable], output: Hashable
    ) -> Callable:
        """Fuse a subgraph of tasks into a single compiled function.

        It is assumed that the function will be called with values corresponding to
        `inputs` in the order they are given.
        """
        if not self._initialized:
            self.initialize_runtime()

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
            tbl.register_func(
                key,
                mlir_func,
                args,
                arg_types=mlir_func.arg_types,
                ret_type=mlir_func.ret_type,
            )

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

        wrapper_func = compile_wrapper(self._engine, mlir_func, constant_vals)

        return wrapper_func
