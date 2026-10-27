"""双击即运行：与 train.py 相同，但坐标通道开启。
双击此文件（而不是 train.py）来跑 CoordConv 实验组。
"""
import os, sys, runpy
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ['train.py', '--coords']
runpy.run_path('train.py', run_name='__main__')
