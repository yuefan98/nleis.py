from nleis import __version__
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nleis",
    version=__version__,
    author="Yuefan Ji and nleis.py developers",
    author_email="yuefan@uw.edu",
    description=("A NLEIS toolbox for impedance.py that provides RC level "
                 "nonlinear equivalent circuit modeling (nECM) and analysis"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://nleispy.readthedocs.io/en/latest/",
    project_urls={
        "Bug Tracker": "https://github.com/yuefan98/nleis.py",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(include=['nleis']),
    package_data={
        "nleis": ["data/*", 'tests/*'],

    },

    python_requires=">=3.8,<3.14",
    install_requires=['altair>=3.0', 'matplotlib>=3.5',
                      'numpy>=1.14', 'scipy>=1.0',
                      'networkx>=2.6.3',
                      'impedance>=1.7.1', 'pandas >= 2.0.2',
                      'threadpoolctl>=3.5.0', 'joblib>=1.4.2', 'SALib>=1.4.8',
                      'tqdm>=4.66.5'],
)
