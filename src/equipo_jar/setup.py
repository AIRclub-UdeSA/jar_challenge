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
        ('share/' + package_name + '/behavior_trees', glob('behavior_trees/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nombre del equipo',
    maintainer_email='equipo@example.com',
    description='Software del equipo para el Challenge JAR 2026.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cobertura = equipo_jar.cobertura:main',
            'deteccion = equipo_jar.deteccion:main',
            'fusion = equipo_jar.fusion:main',
            'mision = equipo_jar.mision:main',
            'parada = equipo_jar.parada:main',
            'registro = equipo_jar.registro:main',
            'localizacion = equipo_jar.localizacion:main',
            'odometria_imu = equipo_jar.odometria_imu:main',
        ],
    },
)
