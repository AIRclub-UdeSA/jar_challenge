from setuptools import setup

package_name = 'jar_supervisor'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='RealBoBots',
    maintainer_email='equipo@example.com',
    description='Supervisor de movimiento: único publicador de /cmd_vel (ADR-0007).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'supervisor = jar_supervisor.supervisor:main',
        ],
    },
)
