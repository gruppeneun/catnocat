#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 18:16:02 2021

@author: Doris Zahnd
This script sorts all image files into three separate batches.
"""

import os
from shutil import copyfile, move
src = '/Users/michael/Documents/Ausbildung/CAS Machine Intelligence/02_Deep_Learning/30_Projekt/bilder'
dst = '/Users/michael/Documents/Ausbildung/CAS Machine Intelligence/02_Deep_Learning/30_Projekt/output'
M = 28482
i = 0
for filename in os.listdir(src):
    if filename.endswith(".jpg"):
        i += 1
        src_file = os.path.join(src, filename)
        print(filename)
        if i < M:
            dst_file = os.path.join(dst, 'batch1', filename)
            move(src_file, dst_file)
        elif i < 2*M:
            dst_file = os.path.join(dst, 'batch2', filename)
            move(src_file, dst_file)
        else:
            dst_file = os.path.join(dst, 'batch3', filename)
            move(src_file, dst_file)
    else:
        continue