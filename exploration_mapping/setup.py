from setuptools import find_packages, setup

package_name = 'exploration_mapping'

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
    maintainer='thor-usr',
    maintainer_email='649365461@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'exploration_map_node = exploration_mapping.exploration_map_node:main',
        'exploration_map_node2 = exploration_mapping.exploration_map_node2:main',
        'exploration_map_node3 = exploration_mapping.exploration_map_node3:main',
        'exploration_map_node4 = exploration_mapping.exploration_map_node4:main',
        'exploration_map_node5 = exploration_mapping.exploration_map_node5:main',
        ],
    },
)
