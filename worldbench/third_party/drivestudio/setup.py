from setuptools import setup, find_packages

setup(
    name='drivestudio',             # 可以自定义
    version='0.1',
    packages=find_packages(),     # 自动查找当前目录下的所有包（含 __init__.py 的目录）
)