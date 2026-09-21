import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'factr_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files live at the repo top level (FACTR_Teleop/launch),
        # two levels up from this setup.py (src/factr_teleop/).
        (os.path.join('share', package_name, 'launch'), glob('../../launch/*.py')),
    ],
    install_requires=['setuptools', 'dynamixel-sdk==4.0.5'],
    zip_safe=True,
    maintainer='Jason Jingzhou Liu and Yulong Li',
    maintainer_email='liujason@cmu.edu',
    description='FACTR low-cost force-feedback teleoperation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'factr_teleop_franka = factr_teleop.factr_teleop_franka_zmq:main',
            'factr_teleop_grav_comp_demo = factr_teleop.factr_teleop_grav_comp_demo:main',
            'factr_rizon_testing = factr_teleop.factr_rizon_teleop:main',
            'factr_joint_pub = factr_teleop.factr_joint_publisher:main',
            'frdb = factr_teleop.factr_rizon_dual_board:main'
        ],
    },
)
