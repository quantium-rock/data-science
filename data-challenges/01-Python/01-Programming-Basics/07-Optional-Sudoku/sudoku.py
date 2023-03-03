# pylint: disable=missing-docstring


def sudoku_validator(grid):

    ln = len(grid)
    sm = 45

    # check rows
    for row in grid:
        if len(set(row)) != ln or sum(row) != sm:
            return False

    # check columns
    for i in range(ln):
        col = []
        for row in grid:
            col.append(row[i])
        if len(set(col)) != ln or sum(col) != sm:
            return False

    # check squares
    k = 3
    for i in range(0, ln, k):
        sq = []
        for j in range(0, ln):
            if len(sq) == ln:
                if len(set(sq)) != ln or sum(sq) != sm:
                    return False
            if j % k == 0:
                sq = grid[j][i:i+k]
            else:
                sq.extend(grid[j][i:i+k])

    return True
