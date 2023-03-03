# pylint: disable=missing-docstring

import os


def start():
    try:
        if os.environ['FLASK_ENV'] == 'development':
            return 'Starting in development mode...'
        elif os.environ['FLASK_ENV'] == 'production':
            return 'Starting in production mode...'
    except:
        return 'Starting in empty mode...'


if __name__ == "__main__":
    print(start())
