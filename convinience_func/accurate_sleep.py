# -*- coding: utf-8 -*-

from platform import python_version_tuple
import time

str_ver = python_version_tuple()
num_ver = float(str_ver[0] + "." + str_ver[1])

if(num_ver >= 3.3):
    #正確な計時
    def accurate_timer(func, dt):
         sttime = time.perf_counter()
         func(dt)
         entime = time.perf_counter()
         return (entime - sttime)
    
    #busy waitを用いた正確なsleep
    def accurate_sleep(dt):
         current_time = time.perf_counter()
         while(time.perf_counter() < current_time + dt):
             pass

else:
    #正確な計時
    def accurate_timer(func, dt):
         sttime = time.clock()
         func(dt)
         entime = time.clock()
         return (entime - sttime)
    
    #busy waitを用いた正確なsleep
    def accurate_sleep(dt):
         current_time = time.clock()
         while(time.clock() < current_time + dt):
             pass
