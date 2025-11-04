from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vision_opencv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'CameraCalibration/yamls'), glob(os.path.join('CameraCalibration/yamls', '*.yaml'))),
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='qing',
    maintainer_email='2596208480@qq.com',
    description='KFS状态识别（Aruco+QR方案）',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'kfs_mapper = PoseSolver.kfs_mapper:main',
            'aruco_detector_node = cv_lib.aruco_detector_node:main', 
            'generate_aruco = cv_lib.generate_aruco:main' ,
            'aruco_feedback_display = cv_lib.aruco_feedback_display:main',
            'qr_kfs_decoder = cv_lib.qr_kfs.qr_kfs_decoder:main', 
            'qr_kfs_generator = cv_lib.qr_kfs.qr_kfs_generator:main', 
        ],
    },
)

