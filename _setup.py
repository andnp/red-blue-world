from setuptools import setup, find_packages

setup(
    name="red_blue_world",
    packages=find_packages(exclude=["tests*"]),
    install_requires=["numpy"],
    version='0.1.0'
)