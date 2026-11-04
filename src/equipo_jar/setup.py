from glob import glob

from setuptools import setup

package_name = 'equipo_jar'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/params', glob('params/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Equipo CHANGUITRON',
    maintainer_email='equipo@example.com',
    description='Software del equipo para el Challenge JAR 2026.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nodo_arranque = equipo_jar.nodo_arranque:main',
            'nodo_equipo = equipo_jar.nodo_equipo:main',
            'nodo_mision = equipo_jar.nodo_mision:main',
            'nodo_pose_inicial = equipo_jar.nodo_pose_inicial:main',
            'nodo_percepcion = equipo_jar.nodo_percepcion:main',
        ],
    },
)
