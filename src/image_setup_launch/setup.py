from setuptools import find_packages, setup

package_name = 'image_setup_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (f'share/{package_name}/launch', ['launch/image_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Elaina',
    maintainer_email='2733813241@qq.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [

        ],
    },
)
