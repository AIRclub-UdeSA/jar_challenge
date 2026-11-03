from glob import glob

from setuptools import find_packages, setup

package_name = 'jar_percepcion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sebastian',
    maintainer_email='2006sebastianm@gmail.com',
    description='Detección de color y de obstáculos no mapeados',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'detector_color = jar_percepcion.detector_color:main',
            'detector_scan = jar_percepcion.detector_scan:main',
            'detector_obstaculos = jar_percepcion.detector_obstaculos:main',
        ],
    },
)
