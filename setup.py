from nleis import __version__
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nleis",
    version= __version__ ,
    author="Yuefan Ji",
    author_email="yuefan@uw.edu",
    description="A NLEIS toolbox for impedance.py that provides RC level nonlinear equivalent modeling and analysis",
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
    packages=setuptools.find_packages(include = ['nleis']),
    package_data={
        "nleis": ["data/*",'nleis_tests/*'],

	},
    
    python_requires=">=3.6",
    install_requires=['altair>=3.0', 'matplotlib>=3.5',
                  'numpy>=1.14', 'scipy>=1.0','impedance>=1.7.1'],
)
