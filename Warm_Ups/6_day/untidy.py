#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

def exampleone():
    """first example of an untidy dataset"""
    example_untitdy = {'country' : ['Afghanistan', 'Afghanistan', 'Afghanistan', 'Afghanistan', 'Brazil', 'Brazil'],
    'year' : [1999, 1999, 2000,2000, 1999, 1999],
    'obser' : ['cases', 'population','cases','population','cases','population'],
    'count': [745,19987071,2666,20595360,37737, 172006362]}
    df_untidy = pd.DataFrame(example_untitdy)
    print(df_untidy)


def exampletwo():
    """second example of an untidy dataset"""
    second_example= {'country':['Afghanistan', 'Afghanistan', 'Brazil', 'Brazil', 'China','China' ],
                     'year':[1999,2000,1999,2000,1999,2000],
                     'rate':['745/19987071','2666/20595360', '37737/172006362', '80488/174504898', '212258/1272915272', '213766/1280428583']}
    df_untidy2 = pd.DataFrame(second_example)
    print(df_untidy2)


def examplethree():
    """third example of an untidy dataset"""
    part1 = {'country':['Afghanistan', 'Brazil', 'China'],
             '1999':[745,37737,212258],
             '2000':[2666, 80488, 213766]}
    part2 = {'country':['Afghanistan', 'Brazil', 'China'],
             '1999':[19987071,172006362,1272915272],
             '2000':[20595360, 174504898, 1280428583]}
    part1 = pd.DataFrame(part1)
    part2 = pd.DataFrame(part2)
    print('table 1')
    print(part1,"\n")
    print('table 2')
    print(part2)




