from setuptools import setup

setup(
    name='PyALMA3',
    version='1.0.0',
    author='Marshall J. Styczinski, Saikiran Tharimena, Daniele Melini, Giorgio Spada, Steven D. Vance',
    author_email='itsmoosh@gmail.com',
    description='Python plAnetary Love nuMbers cALculator',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/drsaikirant88/PyALMA3',
    project_urls={
        'Bug tracker': 'https://github.com/drsaikirant88/PyALMA3/issues',
        'Publication': 'https://doi.org/10.1093/gji/ggac263'
    },
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent'
    ],
    packages=['alma'],
    package_dir={'alma': 'alma'},
    install_requires=[
        'numpy >= 1.26.3',
        'mpmath >= 1.3.0',
        'psutil >= 5.9.7',
        'joblib >= 1.3.2',
        'toml >= 0.10.2'
    ],
    include_package_data=True  # Files to include are listed in MANIFEST.in
)
