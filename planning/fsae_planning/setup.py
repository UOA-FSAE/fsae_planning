import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'fsae_planning'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='UoA FSAE',
    maintainer_email='fsae@auckland.ac.nz',
    description='Simulator path planning: centerline and skidpad planners',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'centerline_planner = fsae_planning.centerline_planner:main',
            'skidpad_planner    = fsae_planning.special_utils.skidpad_planner:main',
            # The kart's cone_mapper_shim (verbatim copy) and the long-retention
            # cone_memory -- see perception.launch.py.
            'cone_mapper_shim   = fsae_planning.cone_mapper_shim:main',
            'cone_memory        = fsae_planning.cone_memory:main',
        ],
    },
)
