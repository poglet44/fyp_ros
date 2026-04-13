from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'sim_tb3_velodyne'


def package_files(directory, install_dir):
    paths = []
    for path, _, filenames in os.walk(directory):
        if filenames:
            files = [os.path.join(path, f) for f in filenames]
            paths.append(
                (
                    os.path.join(
                        'share',
                        package_name,
                        install_dir,
                        os.path.relpath(path, directory)
                    ),
                    files
                )
            )
    return paths


data_files = [
    ('share/ament_index/resource_index/packages',
     ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    (os.path.join('share', package_name, 'config'), glob('config/*.yml')),
]

data_files += package_files('models', 'models')

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sam - Thor-Usr',
    maintainer_email='ninjamanl@hotmail.co.uk',
    description='Local TurtleBot3 simulation package for Velodyne modification',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'pointcloud_passthrough = sim_tb3_velodyne.pointcloud_passthrough:main',
        'ogm_node = sim_tb3_velodyne.ogm_node:main',
        "ogm_from_cloud_node = sim_tb3_velodyne.ogm_from_cloud_node:main",
        "local_ogm_node = sim_tb3_velodyne.local_ogm_node:main",
        "global_ogm_stitcher_node = sim_tb3_velodyne.global_ogm_stitcher_node:main",
        'occupancy_grid_inflater = sim_tb3_velodyne.occupancy_grid_inflater:main',
        ]},
)