from setuptools import find_packages, setup

package_name = 'fsae_planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/launch_planning.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tommy',
    maintainer_email='tommy@todo.todo',
    description='Formula Student Driverless planning and control stack',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'planner_node     = fsae_planning.planner_node:main',
            'perception_node  = fsae_planning.track_utils.perception_node:main',
            'control_node     = fsae_planning.track_utils.control_node:main',
            'integration_node = fsae_planning.integration_node:main',
        ],
    },
)
