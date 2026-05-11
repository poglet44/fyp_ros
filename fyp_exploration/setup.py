from setuptools import setup

package_name = 'fyp_exploration'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/frontier_detector.launch.py']),
        ('share/' + package_name + '/config', ['config/frontier_detector.yaml']),
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
        ],
    },
)
