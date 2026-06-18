from setuptools import find_packages, setup

package_name = 'bc'

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
    maintainer='Jason Jingzhou Liu and Yulong Li',
    maintainer_email='liujason@cmu.edu',
    description='Data collection for Behavior Cloning and policy roll out',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_record = bc.data_record:main',
            'data_sync = bc.data_sync:main',
            'replay_traj = bc.replay_traj:main',
            'franka_bridge = bc.franka_bridge:main',
            'policy_rollout = bc.policy_rollout:main',
        ],
    },
)
