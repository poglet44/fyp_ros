from glob import glob
import os

from setuptools import setup

package_name = 'fyp_exploration'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Install all launch files.
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),

        # Install all YAML config files.
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),

        # Install RViz configs if present.
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sam',
    maintainer_email='sam@example.com',
    description='Frontier detection and visualisation for autonomous exploration.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'frontier_detector = fyp_exploration.frontier_detector:main',
            'exploration_nav2_executor = fyp_exploration.exploration_nav2_executor:main',
        ],
    },
)
