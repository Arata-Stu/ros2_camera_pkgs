from setuptools import setup
import os
from glob import glob

package_name = 'segmentation_node'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        
        # package.xml をインストール
        (os.path.join('share', package_name), ['package.xml']),
        # launch/config フォルダを含める
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='jetson@todo.todo',
    description='Segmentation Node for ROS2 using MobileNetV3 and FastSCNN',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segmentation_node = segmentation_node.segmentation_node:main'
        ],
    },
)