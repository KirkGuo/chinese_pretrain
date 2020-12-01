import requests
import os
import time
import urllib.parse
import string
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

from multiprocessing import Process, Manager

curr_time = int(time.time())

def get_root_page(root_url='http://www3.zzu.edu.cn/qtss/zzjpoem1.dll/query'):

    root_page = requests.get(root_url)

    root_soup = BeautifulSoup(root_page.content.decode('gbk'), 'html.parser')

    names = root_soup.table.find_all('font')
    hrefs = root_soup.table.find_all('a')

    names = list(names)[2:]
    hrefs = list(hrefs)[2:]
    assert len(names) == len(hrefs)
    return names, hrefs


def worker_func(param):
    root_path, names, hrefs, indexes = param
    error_urls = []
    for idx in tqdm(indexes):
        name, href = names[idx].text, urllib.parse.quote(hrefs[idx].attrs['href'].encode('gbk'), safe=string.printable)
        path = os.path.join(root_path, name)
        if not os.path.exists(path):
            os.makedirs(path)
        poems_page = requests.get(href)
        poems_soup = BeautifulSoup(poems_page.content.decode('gbk'), 'html.parser')
        poem_hrefs = poems_soup.select('body>div:nth-child(1)>center>table>tr:nth-child(3)')[0].find_all('a')
        for poem_href in poem_hrefs:
            poem_href, poem_name = urllib.parse.quote(poem_href.attrs['href'].encode('gbk'), safe=string.printable), poem_href.text
            poem_page = requests.get(poem_href)
            try:
                poem_soup = BeautifulSoup(poem_page.content.decode('gbk'), 'html.parser')
                poem = poem_soup.select('body>div:nth-child(1)>center>table>tr:nth-child(5)>td>p>font')[0].text
                poem = re.sub('\xa0', '', poem)
                with open(os.path.join(path, f'{poem_name}.txt'), 'w') as f:
                    f.write(poem)
            except UnicodeDecodeError:
                print(poem_href)
                error_urls.append(poem_href)
    return error_urls


if __name__ == '__main__':
    root_path = r'D:\dataset\poem\Tang'
    names, hrefs = get_root_page()
    indexes = range(len(names))
    error_urls = worker_func((root_path, names, hrefs, indexes))

