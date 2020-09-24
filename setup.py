from setuptools import setup, find_packages

setup(
    name="ir-metrics",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
    ]
)
