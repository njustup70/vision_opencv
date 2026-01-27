from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'comm_ws'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='qing',
    maintainer_email='2596208480@qq.com',
    description='KFS状态识别（Aruco+QR方案）',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'camera_node = test.qr_kfs.camera_node:main',
            'qr_detect_node = test.qr_kfs.qr_detect_node:main',
        ],
    },
)
