# pylint: disable=missing-docstring

import sys


def full_name(first_name, last_name):
    """returns the full name"""
    name = f"{first_name.capitalize()} {last_name.capitalize()}"
    return name.strip()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("You must provide a first name and a last name as arguments!")
    else:
        print(f"Hello {full_name(sys.argv[1], sys.argv[2])}")
