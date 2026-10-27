"""双击即运行：测试 CoordConv 实验组权重（model_coords_True.pth）。"""
import os, sys, runpy
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ['test.py', '--coords']
runpy.run_path('test.py', run_name='__main__')
