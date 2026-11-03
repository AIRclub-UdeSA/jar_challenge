from glob import glob

from setuptools import find_packages, setup

package_name = 'jar_localizacion'

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
    description='Localización en el mapa con filtro de partículas propio',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'campo_verosimilitud = jar_localizacion.campo_verosimilitud:main',
            'localizador = jar_localizacion.localizador:main',
            'medir_odometria = jar_localizacion.medir_odometria:main',
            'odom_corregida = jar_localizacion.odom_corregida:main',
            'publicar_mapa = jar_localizacion.publicar_mapa:main',
        ],
    },
)
