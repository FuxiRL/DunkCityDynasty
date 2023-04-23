'''
Descripttion: 
version: 
Author: Simsimi
Date: 2023-04-11 19:47:37
'''
import time


def sleep(sec):
    """custom sleep function
    """
    starttime = time.time()
    while True:
        for _ in range(1000):
            pass
        if time.time() - starttime > sec:
            return
