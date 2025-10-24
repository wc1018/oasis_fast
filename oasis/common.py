import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial, wraps
from time import time
from typing import Any, Callable

import numpy

# Gravitational constant
G_gravity: float = 4.3e-09     # Mpc (km/s)^2 / M_sun


@dataclass(frozen=True)
class F:
    """
    """
    HEADER: str = "\033[35m "
    OKBLUE: str = "\033[34m "
    OKCYAN: str = "\033[36m "
    OKGREEN: str = "\033[32m "
    WARNING: str = "\033[33m "
    FAIL: str = "\033[31m "
    ENDC: str = "\033[0m "
    BOLD: str = "\033[1m "
    UNDERLINE: str = "\033[4m "
    BULLET: str = "\u25CF "


def timer(
    procedure: Callable = None, *,
    fancy: bool = True,
    off: bool = False
) -> Callable:
    """
    Decorator that prints the procedure's execution time.

    Parameters
    ----------
    procedure : Callable
        Any callable.
    fancy : bool, optional
        Prints timer message using colours, by default False.
    off : bool, optional
        Turns off the timer, by default False.

    Returns
    -------
    Callable
        Returns callable object/return value.
    """
    if procedure is None:
        return partial(timer, fancy=fancy, off=off)

    @wraps(procedure)
    def wrapper(*args, **kwargs):
        fmt: str = '%Y-%m-%d %H:%M:%S'
        start = datetime.now()
        t_start = time()

        return_value = procedure(*args, **kwargs)

        finish = datetime.now()
        dt = timedelta(seconds=time()-t_start)
        if fancy:
            label = f"\t{F.BOLD}Process:{F.ENDC} " \
                f"{F.FAIL}{procedure.__name__}{F.ENDC} \n" \
                f"Start:  {F.OKBLUE}{start.strftime(fmt)}{F.ENDC} \n" \
                f"Finish: {F.OKBLUE}{finish.strftime(fmt)}{F.ENDC} \n" \
                f"{F.BULLET}{F.BOLD}{F.OKGREEN} Elapsed time:{F.ENDC} " \
                f"{F.WARNING}{dt}{F.ENDC}"
        else:
            label = f"\t Process: {procedure.__name__} \n" \
                f"Start:  {start.strftime(fmt)} \n" \
                f"Finish: {finish.strftime(fmt)} \n" \
                f"{F.BULLET} Elapsed time: {dt}"
        if not off:
            print(label)

        return return_value

    return wrapper


def get_np_unit_dytpe(num: Any) -> numpy.dtype:
    """Determines the minimum unsigned integer type to represent `num`.

    Parameters
    ----------
    num : Any
        Numerical value.

    Returns
    -------
    numpy.dtype
        Numpy data type class.

    Raises
    ------
    TypeError
        If `num` is not an integer or it is a negative value.
    OverflowError
        If `num` cannot be represented by any 16, 32 or 64 bit unsigned integer.
    """
    if num < 0 or type(num) in [float]:
        raise TypeError

    np_unit_dtypes = numpy.array([numpy.uint16, numpy.uint32, numpy.uint64])
    check: list[bool] = [num < numpy.iinfo(
        item).max for item in np_unit_dtypes]
    if any(check):
        return np_unit_dtypes[numpy.argmax(check)]
    else:
        raise OverflowError


def mkdir(path: str, verbose: bool = False) -> int:
    """Checks if a path exists and creates a directory in path if it does not.

    Parameters
    ----------
    path : str
        Path where directory should exist. 
    verbose : bool, optional
        Print info on directory creation process, by default False.

    Returns
    -------
    int 
        Returns 1 if the directory was created successfully.

    Raises
    ------
    FileNotFoundError
        If the path to directory does not exist.
    """
    abspath: str = os.path.abspath(path)
    isdir: bool = os.path.isdir(abspath)
    if not isdir:
        try:
            os.makedirs(os.path.abspath(path), exist_ok=True)
            isdir = os.path.isdir(abspath)
            if isdir and verbose:
                print(f"Directory created at {abspath}")
            return 1
        except:
            msg = f"Directory could not be created at {abspath}"
            raise FileNotFoundError(msg)
    else:
        if verbose:
            print(f"Directory exists at {abspath}")
        return 1


###
