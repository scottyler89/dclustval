from setuptools import setup, find_packages

setup(
    name="dclustval",
    version="0.1.0",
    description="A package for performing dense cluster validation",
    packages=find_packages(),
    install_requires=[
        # add your package dependencies here, e.g.
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "statsmodels",
	"networkx",
        "count_split",
        "scikit_learn"
    ],
    extras_require={
        'docs': [
            'm2r2'
        ]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires='>=3.6',
)

