import os
import pandas as pd


class Olist:
    def get_data(self):
        """
        This function returns a Python dict.
        Its keys should be 'sellers', 'orders', 'order_items' etc...
        Its values should be pandas.DataFrames loaded from csv files
        """
        # Hints 1: Build csv_path as "absolute path" in order to call this method from anywhere.
            # Do not hardcode your path as it only works on your machine ('Users/username/code...')
            # Use __file__ instead as an absolute path anchor independant of your usename
            # Make extensive use of `breakpoint()` to investigate what `__file__` variable is really
        # Hint 2: Use os.path library to construct path independent of Mac vs. Unix vs. Windows specificities
        root_path = os.path.dirname(os.path.dirname(__file__))
        csv_path = os.path.join(root_path, 'data', 'csv')
        files_names = os.listdir(csv_path)
        key_names = [ x.replace('olist_','').replace('_dataset','').replace('.csv','') for x in files_names ]
        data = {}
        for key, file in zip(key_names, files_names):
            data[key] = pd.read_csv(os.path.join(csv_path, file))
        return data

    def ping(self):
        """
        You call ping I print pong.
        """
        print("pong")
