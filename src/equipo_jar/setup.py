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
    maintainer='Nombre del equipo',
    maintainer_email='equipo@example.com',
    description='Software del equipo para el Challenge JAR 2026.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'principal = equipo_jar.principal:main',
            'nodo_equipo = equipo_jar.nodo_equipo:main',
            'detector_obstaculos = equipo_jar.detector_obstaculos:main',
            'campo_verosimilitud = equipo_jar.campo_verosimilitud:main',
            'localizador = equipo_jar.localizador:main',
            'evasor = equipo_jar.evasor:main',
            'detector_scan = equipo_jar.detector_scan:main',
            'gestor_victimas = equipo_jar.gestor_victimas:main',
        ],
    },
)
