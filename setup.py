from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
        name='mujoco_utils',
        version='1.0.0',
        description='Framework and utilities for implementing and interfacing with MuJoCo and MuJoCo-XLA environments.',
        long_description=readme,
        url='https://github.com/Co-Evolve/mujoco-utils',
        license=license,
        packages=find_packages(exclude=('tests', 'docs')),
        install_requires=required
        )
