from setuptools import find_packages, setup

package_name = 'depth_camera_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Elaina',
    maintainer_email='2733813241@qq.com',
    description='TODO: Package description',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'depth_camera_node = depth_camera_pkg.depth_camera_node:main',
        ],
    },
)
