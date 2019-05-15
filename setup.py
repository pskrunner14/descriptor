from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='descriptor',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1',

    description='Image Captioning Model',
    long_description=long_description,

    # The project's main homepage.
    url='http://github.com/pskrunner14/descriptor',
    author='Prabhsimran Singh',
    author_email='pskrunner14@gmail.com',
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Research',
        'Topic :: Software Development :: NLP',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6'
    ],

    # What does your project relate to?
    keywords='descriptor image-captioning encoder-decoder autoencoder cnn-encoder rnn-decoder',

    packages=find_packages(exclude=['dev', 'docs', 'tests']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'torch',
        'torchvision',
        'torchtext',
        'numpy'
    ],
    include_package_data=True,
    zip_safe=False
)