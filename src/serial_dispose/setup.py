from setuptools import setup, find_packages

setup(
    name='serial_dispose',
    version='0.0.1',
    packages=find_packages(),  # 自动找到 serial_dispose 子目录和 get_dispose_serial
    install_requires=['setuptools', 'pyserial'],  # 如果你要用 serial 库
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'serial_dispose = serial_dispose.serial_dispose:main',
        ],
    },
)