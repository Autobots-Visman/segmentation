from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import find_packages, setup

# http://docs.ros.org/en/jade/api/catkin/html/user_guide/setup_dot_py.html
setup(
    **generate_distutils_setup(
        packages=find_packages("src"),
        package_dir={"": "src"},
    )
)
