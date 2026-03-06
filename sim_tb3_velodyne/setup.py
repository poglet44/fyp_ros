from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'sim_tb3_velodyne'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),

        (os.path.join('share', package_name, 'worlds'),
         glob('worlds/*')),

        (os.path.join('share', package_name, 'urdf'),
         glob('urdf/*')),

        (os.path.join('share', package_name, 'models', 'turtlebot3_burger'),
         glob('models/turtlebot3_burger/*')),

        (os.path.join('share', package_name, 'models', 'turtlebot3_common'),
         ['models/turtlebot3_common/model.config']),

        (os.path.join('share', package_name, 'models', 'turtlebot3_common', 'meshes'),
         glob('models/turtlebot3_common/meshes/*.dae')),

        (os.path.join('share', package_name, 'models', 'turtlebot3_common', 'meshes', 'bases'),
         glob('models/turtlebot3_common/meshes/bases/*')),

        (os.path.join('share', package_name, 'models', 'turtlebot3_common', 'meshes', 'sensors'),
         glob('models/turtlebot3_common/meshes/sensors/*')),

        (os.path.join('share', package_name, 'models', 'turtlebot3_common', 'meshes', 'wheels'),
         glob('models/turtlebot3_common/meshes/wheels/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sam - Thor-Usr',
    maintainer_email='ninjaman1@hotmail.co.uk',
    description='Local TurtleBot3 simulation package for Velodyne modification',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)