import os
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.rst')).read()
except IOError:
    README = ''

setup(
    name="ir-metrics",
    use_scm_version=True,
    description='The most common information retrieval (IR) metrics',
    long_description=README,
    url="https://github.com/kqf/ir-metrics",
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
    ]
)
