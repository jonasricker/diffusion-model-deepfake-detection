import functools
import inspect
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from hashlib import sha1
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Union

import cytoolz
from boltons.funcutils import wraps
from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_with_cuda_env(func, *args, device: int, **kwargs) -> Any:
    """Set CUDA_VISIBLE_DEVICES to desired device before running a function and reset afterwards."""
    before = os.getenv("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    output = func(*args, **kwargs)
    if before is None:
        os.unsetenv("CUDA_VISIBLE_DEVICES")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = before
    return output


def parallel_map(
    func: Callable,
    *iterables: Iterable,
    func_kwargs: Optional[Dict[str, Any]] = None,
    num_workers: int = 1,
    mode: Literal["multiprocessing", "multithreading"] = "multiprocessing",
    gpu_ids: Optional[Union[int, List[int]]] = None,
) -> List[Any]:
    """Similar to map, but can be used with multithreading/multiprocessing and across multiple GPUs."""
    # validate
    if not all(
        len(list(iterable)) == len(list(iterables[0])) for iterable in iterables
    ):
        raise ValueError("All iterables must have equal length.")
    if func_kwargs is None:
        func_kwargs = {}
    if mode not in ["multiprocessing", "multithreading"]:
        raise NotImplementedError(
            f"Mode {mode} is not supported, use multiprocessing or multithreading."
        )

    use_gpu = gpu_ids is not None
    if use_gpu:
        if num_workers > 1 and mode == "multithreading":
            raise ValueError("Use multiprocessing mode for parallel GPU use.")
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        num_workers = len(gpu_ids) * num_workers
        gpu_pool = sorted(gpu_ids * num_workers)

    # sequential execution for single worker or when debugging, parallel execution otherwise
    pbar = tqdm(total=len(list(iterables[0])), desc=func.__name__)
    if num_workers == 1 or "DEBUG" in os.environ:
        output = []
        for args in zip(*iterables):
            if use_gpu:
                output.append(
                    run_with_cuda_env(func, *args, device=gpu_ids[0], **func_kwargs)
                )
            else:
                output.append(func(*args, **func_kwargs))
            pbar.update()
    else:
        executor = (
            ProcessPoolExecutor if mode == "multiprocessing" else ThreadPoolExecutor
        )
        with executor(max_workers=num_workers) as pool:
            futures = []
            for args in zip(*iterables):
                if use_gpu and len(gpu_pool) == 0:
                    done, _ = wait(
                        [future for future in futures if future.device is not None],
                        return_when="FIRST_COMPLETED",
                    )
                    for future in done:
                        gpu_pool.append(future.device)
                        future.device = None
                if use_gpu:
                    gpu = gpu_pool.pop(0)
                    future = pool.submit(
                        run_with_cuda_env, func, *args, device=gpu, **func_kwargs
                    )
                    future.device = gpu
                else:
                    future = pool.submit(func, *args, **func_kwargs)
                future.add_done_callback(
                    lambda f: print(f.exception()) if f.exception() else None
                )
                future.add_done_callback(lambda _: pbar.update())
                futures.append(future)
        if any(future.exception() for future in futures):
            raise Exception(
                "One or more futures raised exceptions. Set num_workers to 1 "
                "or set the DEBUG environment variable to get a detailed stacktrace."
            )
        output = [future.result() for future in futures]
    return output


def with_caching(keys: List[str]):
    """Decorator to enable caching on arbitrary functions."""

    def decorator(func):
        @wraps(func, expected=[("cache_dir", None), ("overwrite", False)])
        def wrapper(*args, **kwargs):
            args, cache_dir, overwrite = args[:-2], args[-2], args[-1]
            if cache_dir is None:
                raise ValueError("'cache_dir' must not be None.")

            # construct complete arguments from args, kwargs, and defaults
            ba = inspect.signature(func).bind(*args, **kwargs)
            ba.apply_defaults()
            arguments = ba.arguments

            # skip caching when debugging
            if "DEBUG" in os.environ:
                return func(*args, **kwargs)

            # construct key and compute hash
            hash = sha1(
                args_to_bytes(**{k: v for k, v in arguments.items() if k in keys})
            ).hexdigest()

            # load from cache or compute and cache
            cache_dir.mkdir(exist_ok=True, parents=True)
            cache_file = cache_dir / f"{hash}.pickle"
            if cache_file.exists() and not overwrite:
                with open(cache_file, "rb") as f:
                    output = pickle.load(f)
            else:
                output = func(*args, **kwargs)
                with open(cache_file, "wb") as f:
                    pickle.dump(output, f)
            return output

        return wrapper

    return decorator


def args_to_bytes(*args, **kwargs) -> bytes:
    """Convert function arguments to hash for the purpose of caching."""
    b = b""
    for arg in args:
        b += type(arg).__name__.encode()

        # builtins
        if isinstance(arg, str):
            b += arg.encode()
        elif isinstance(arg, (int, float, complex)):
            b += str(arg).encode()
        elif isinstance(arg, (list, tuple)):
            b += args_to_bytes(*arg)
        elif isinstance(arg, dict):
            for k, v in arg.items():
                b += args_to_bytes(k)
                b += args_to_bytes(v)

        # third party
        elif isinstance(arg, Path):
            b += str(arg.resolve()).encode()
        elif type(arg).__name__ == "ndarray":
            b += arg.tobytes() + str(arg.shape).encode()
        elif type(arg).__name__ == "Tensor":
            b += args_to_bytes(arg.numpy())

        # functions
        elif isinstance(arg, functools.partial):
            b += (
                arg.func.__name__.encode()
                + args_to_bytes(arg.args)
                + args_to_bytes(arg.keywords)
            )
        elif isinstance(arg, cytoolz.functoolz.Compose):
            b += args_to_bytes(arg.first)
            for func in arg.funcs:
                b += args_to_bytes(func)
        elif inspect.isroutine(
            arg
        ):  # not able to detect changes in nested function calls TODO use inspect to cache all attributes of a class
            name = arg.__name__
            if name == "<lambda>":
                b += "".join(inspect.getsource(arg).split()).encode()
            else:
                b += name.encode()
        else:
            raise NotImplementedError(
                f"No conversion from {type(arg)} to bytes defined."
            )
    for k, v in kwargs.items():
        b += args_to_bytes(k) + args_to_bytes(v)
    return b
