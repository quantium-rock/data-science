# pylint: disable=no-value-for-parameter
"""
Client of the Wagon OpenGraph API
"""
import ast
import requests

def fetch_metadata(url):
    """
    Return a dictionary of OpenGraph metadata found in HTML of given url
    """
    req = requests.get('https://opengraph.lewagon.com/?url='+url)
    if req.status_code != 200:
        return None
    req = req.content.decode('UTF-8')
    req = ast.literal_eval(req)
    return {'title': req['data']['site_name'], 'description': req['data']['description']}


# To manually test, please uncomment the following lines and run `python opengraph.py`:
# import pprint
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(fetch_metadata("https://www.lewagon.com"))
