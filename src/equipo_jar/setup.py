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
        ('share/' + package_name + '/params', glob('params/*.rviz')),
        ('share/' + package_name + '/behavior_trees', glob('behavior_trees/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Stack&Queue',
    maintainer_email='equipo@example.com',
    description='Exploración autónoma y detección de víctimas de Stack&Queue.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'ros2launch.option': [
            'stackqueue_log = equipo_jar.registro_jar:RegistroJarOption',
        ],
        'console_scripts': [
            'exploracion = equipo_jar.exploracion:main',
            'deteccion_victimas = equipo_jar.deteccion_victimas:main',
        ],
    },
)
