# pylint: disable=missing-module-docstring,missing-function-docstring,eval-used
import sys


def main():
    ops = {
        '+': lambda x, y: x+y,
        '-': lambda x, y: x-y,
        '*': lambda x, y: x*y,
        '/': lambda x, y: x/y,
        '//': lambda x, y: x//y,
        '%': lambda x, y: x % y,
        '**': lambda x, y: x**y
    }
    """Implement the calculator"""
    return ops[sys.argv[2]](int(sys.argv[1]), int(sys.argv[3]))


if __name__ == "__main__":
    print(main())
