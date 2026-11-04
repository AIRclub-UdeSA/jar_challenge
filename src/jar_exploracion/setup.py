from setuptools import find_packages, setup

package_name = 'jar_exploracion'

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
    maintainer='sebastian',
    maintainer_email='2006sebastianm@gmail.com',
    description='Grilla de cobertura y exploración autónoma del mapa con A*',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'explorador = jar_exploracion.explorador:main',
            'grilla_cobertura = jar_exploracion.grilla_cobertura:main',
        ],
    },
)
