from setuptools import setup

package_name = 'cameras'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jason Jingzhou Liu and Yulong Li',
    maintainer_email='liujason@cmu.edu',
    description='Nodes for running cameras.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'zed = cameras.zed:main',
            'realsense = cameras.realsense:main',
        ],
    },
)
