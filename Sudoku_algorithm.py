import time
import numpy as np
from itertools import product
from typing import List 

'''
This solver was taken from https://www.cs.mcgill.ca/~aassaf9/python/sudoku.txt under the GNU General Public License.
It expects input in the form of a 2D array, and will return the answer as a 2D array. If it is unsolvable, it will
raise an exception.
'''


def solve_sudoku(size: tuple, grid: list) -> list:
    """
    Solve a Sudoku puzzle using the exact cover algorithm.

    Args:
        size (tuple of int): The size of the Sudoku grid as (rows, columns). 
                             For a standard 9x9 Sudoku, this would be (3, 3).
        grid (list of lists of int): The initial Sudoku grid as a 2D list. 
                                     Empty cells should be represented by 0.

    Yields:
        list of lists of int: The solved Sudoku grid. Each sublist represents a row.
    """
    R, C = size
    N = R * C
    X = ([("rc", rc) for rc in product(range(N), range(N))] +
         [("rn", rn) for rn in product(range(N), range(1, N + 1))] +
         [("cn", cn) for cn in product(range(N), range(1, N + 1))] +
         [("bn", bn) for bn in product(range(N), range(1, N + 1))])

    Y = dict()
    for r, c, n in product(range(N), range(N), range(1, N + 1)):
        b = (r // R) * R + (c // C)  # Box number
        Y[(r, c, n)] = [
            ("rc", (r, c)),
            ("rn", (r, n)),
            ("cn", (c, n)),
            ("bn", (b, n))]
    X, Y = exact_cover(X, Y)
    for i, row in enumerate(grid):
        for j, n in enumerate(row):
            if n:
                select(X, Y, (i, j, n))
    for solution in solve(X, Y, []):
        for (r, c, n) in solution:
            grid[r][c] = n
        yield grid


def exact_cover(X: list, Y: dict) -> tuple:
    """
    Create the exact cover problem representation.

    Args:
        X (list): The set of elements to cover, represented as a list of strings or tuples.
        Y (dict): The collection of subsets of X. Keys are (row, column, number) tuples,
                  values are lists of constraints.

    Returns:
        tuple: The exact cover problem as (X, Y), where X is now a dict and Y remains unchanged.
               X's keys are the original elements, values are sets of rows that cover the element.
    """
    X = {j: set() for j in X}
    for i, row in Y.items():
        for j in row:
            X[j].add(i)
    return X, Y


def solve(X: dict, Y: dict, solution: list) -> list:
    """
    Solve the exact cover problem using recursive backtracking.

    Args:
        X (dict): The remaining elements to be covered. Keys are constraints, 
                  values are sets of rows that satisfy the constraint.
        Y (dict): The remaining subsets. Keys are (row, column, number) tuples,
                  values are lists of constraints satisfied by this tuple.
        solution (list): The current partial solution, as a list of (row, column, number) tuples.

    Yields:
        list: Complete solutions to the exact cover problem, each a list of (row, column, number) tuples.
    """
    if not X:
        yield list(solution)
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution.append(r)
            cols = select(X, Y, r)
            for s in solve(X, Y, solution):
                yield s
            deselect(X, Y, r, cols)
            solution.pop()


def select(X: dict, Y: dict, r: tuple) -> list:
    """
    Select a row and update the problem accordingly.

    Args:
        X (dict): The remaining elements to be covered.
        Y (dict): The remaining subsets.
        r (tuple): The row to select, as a (row, column, number) tuple.

    Returns:
        list: The columns removed during selection, each a set of rows.
    """
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols


def deselect(X: dict, Y: dict, r: tuple, cols: list) -> None:
    """
    Deselect a row and restore the problem to its previous state.

    Args:
        X (dict): The remaining elements to be covered.
        Y (dict): The remaining subsets.
        r (tuple): The row to deselect, as a (row, column, number) tuple.
        cols (list): The columns removed during selection, each a set of rows.

    Returns:
        None
    """
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)


def solve_wrapper(squares_num_array: str) -> tuple:
    """
    Wrapper function to solve a Sudoku puzzle.

    Args:
        squares_num_array (str): A string representation of the Sudoku puzzle,
                                 where '0' represents empty cells. The string
                                 should be 81 characters long for a 9x9 grid.

    Returns:
        tuple: (solved_puzzle, solve_time)
               solved_puzzle (str or None): A string representation of the solved Sudoku,
                                            or None if unsolvable. The string is 81 characters
                                            long, representing the grid row by row.
               solve_time (str or None): A string indicating the time taken to solve
                                         (e.g., "Solved in 0.1234s"), or None if unsolvable.
    """
    if squares_num_array.count('0') >= 80:
        return None, None

    start = time.time()

    # convert string to 9x9 array
    arr = []
    for i in squares_num_array:
        arr.append(int(i))

    arr = np.array(arr, dtype=int)
    arr = np.reshape(arr, (9, 9))
    try:
        ans = list(solve_sudoku(size=(3, 3), grid=arr))[0]
        s = ""
        for a in ans:
            s += "".join(str(x) for x in a)
        return s, "Solved in %.4fs" % (time.time() - start)
    except:
        return None, None
