# pylint: disable=missing-docstring

def sudoku_solver(grid):

    def check_grid(grid):
        if type(grid) != list:
            return False
        if len(grid) != 9:
            return False
        for row in grid:
            if len(row) != 9:
                return False
        return True

    if not check_grid(grid):
        return 'invalid grid'

    su = grid
    ln = len(su)
    sq = int(ln ** 0.5)

    def xy(su):
        for i in range(ln):
            for j in range(ln):
                if su[i][j] == 0:
                    return (i, j)
        return None

    def check(su, n, ij):
        # check rows
        for i in range(ln):
            if su[ij[0]][i] == n and ij[1] != i:
                return False
        # check columns
        for i in range(ln):
            if su[i][ij[1]] == n and su[0] != i:
                return False

        # check squares
        for i in range(ij[0]//sq*sq, ij[0]//sq*sq+sq):
            for j in range(ij[1]//sq*sq, ij[1]//sq*sq+sq):
                if su[i][j] == n and (i, j) != ij:
                    return False

        return True

    def solve(su):

        ij = xy(su)
        if not ij:
            return True

        for n in range(1, ln+1):
            if check(su, n, ij):
                su[ij[0]][ij[1]] = n
                if solve(su):
                    return True
                su[ij[0]][ij[1]] = 0

        return False

    solve(su)

    return su
