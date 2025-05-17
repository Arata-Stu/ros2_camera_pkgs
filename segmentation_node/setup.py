from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'segmentation_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament index 用
        ('share/ament_index/resource_index/packages',
            [os.path.join('resource', package_name)]),
        # package.xml をインストール
        (os.path.join('share', package_name), ['package.xml']),
        # launch/config フォルダを含める
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'torchvision',
        'torch-geometric',
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='segmentation_node node for processing Camera data',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            # 実行可能スクリプトを定義
            'segmentation_node = segmentation_node.segmentation_node:main',
        ],
    },
)