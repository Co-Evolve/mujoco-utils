from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='mujoco_utils',
    version='0.2.0',
    description='Utilities for interfacing with MuJoCo through dm_control.',
    long_description=readme,
    url='https://github.com/Co-Evolve/mujoco-utils',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=required
)
