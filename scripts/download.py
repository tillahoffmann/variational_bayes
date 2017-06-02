#!/usr/bin/env python

import logging
import re
import urllib
from datetime import datetime
from argparse import ArgumentParser
import os
from time import sleep
import requests
import numpy as np
from tqdm import tqdm
import mechanicalsoup

logger = logging.getLogger(__name__)


def parse_date(s):
    """
    Parse a date string.
    """
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise TypeError(msg)


def download_yahoo(symbol, date_from, date_to, filename=None):
    if filename and os.path.exists(filename):
        with open(filename) as fp:
            text = fp.read()
            logging.info("loaded %d bytes from '%s'", len(text), filename)
            return text, True

    # Open the main website so we can get the crumb
    browser = mechanicalsoup.Browser()
    url = 'https://finance.yahoo.com/quote/%s/history' % symbol
    response = browser.get(url)
    response.raise_for_status()

    logger.info("received %d bytes from '%s'", len(response.text), url)

    # Find the crumb in the source
    crumb = re.search('"CrumbStore":{"crumb":"(.*?)"', response.text)
    assert crumb, "could not find crumb for authentication"
    crumb = crumb.group(1)

    # Perform the download using the same browser and the crumb
    url = "https://query1.finance.yahoo.com/v7/finance/download/%s?" % symbol
    parameters = {
        'period1': int(date_from.timestamp()),
        'period2': int(date_to.timestamp()),
        'interval': '1d',
        'events': 'history',
        'crumb': crumb
    }

    url += urllib.parse.urlencode(parameters)
    response = browser.get(url)
    response.raise_for_status()

    logger.info("received %d bytes from '%s'", len(response.text), url)

    if filename:
        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)
        with open(filename, 'w') as fp:
            fp.write(response.text)

    return response.text, False


def download_google(symbol, date_from, date_to, filename=None):
    if filename and os.path.exists(filename):
        with open(filename) as fp:
            text = fp.read()
            logging.info("loaded %d bytes from '%s'", len(text), filename)
            return text, True

    # http://www.google.co.uk/finance/historical?q=FOXA&startdate=Jan+1%2C+2007&enddate=Jan+1%2C+2017&output=csv
    url = "http://www.google.co.uk/finance/historical?" + urllib.parse.urlencode({
        'q': symbol,
        'startdate': date_from.strftime('%Y-%m-%d'),
        'enddate': date_to.strftime('%Y-%m-%d'),
        'output': 'csv'
    })

    browser = mechanicalsoup.Browser()
    response = browser.get(url)
    response.raise_for_status()

    logger.info("received %d bytes from '%s'", len(response.text), url)

    if filename:
        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)
        with open(filename, 'w') as fp:
            fp.write(response.text)

    return response.text, False


def __main___():
    parser = ArgumentParser(__name__)
    parser.add_argument('--output', '-o', help='output directory', default='.')
    parser.add_argument('--file', '-f', help='file containing a list of ticker symbols')
    parser.add_argument('--log', '-l', help='log file')
    parser.add_argument('--retries', '-r', help='number of retries', type=int, default=3)
    parser.add_argument('source', help="data source to download from", choices=['yahoo', 'google'])
    parser.add_argument('date_from', type=parse_date, help='start date')
    parser.add_argument('date_to', type=parse_date, help='end date')
    parser.add_argument('symbols', nargs='*', help='ticker symbols')
    args = parser.parse_args()

    if args.log:
        logging.basicConfig(filename=args.log, level=logging.INFO)

    if args.file:
        with open(args.file) as fp:
            symbols = [line for line in fp.readlines() if line.strip()]
    else:
        symbols = args.symbols

    # Get the download method
    if args.source == 'yahoo':
        _download = download_yahoo
    elif args.source == 'google':
        _download = download_google
    else:
        raise KeyError(args.source)

    for symbol in tqdm(symbols):
        symbol = symbol.strip()
        filename = os.path.join(args.output, symbol + '.csv')
        num_tries = 0
        success = False
        while num_tries < args.retries and not success:
            try:
                _, cached = _download(symbol, args.date_from, args.date_to, filename)
                success = True
            except requests.exceptions.HTTPError as ex:
                logger.error(ex)
                num_tries += 1
                # Don't retry missing files
                if ex.response.status_code == 404:
                    break

            if not success or not cached:
                sleep((num_tries + 1) * (3 + np.random.gamma(2)))

        if not success:
            message = "exceeded %d retries for symbol %s between %s and %s" % (
                args.retries, symbol, args.date_from, args.date_to
            )
            logger.critical(message)


if __name__ == '__main__':
    __main___()
