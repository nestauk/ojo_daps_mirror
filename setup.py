from setuptools import setup
from setuptools import find_namespace_packages
from ojd_daps import __version__, __basedir__

version = ''.join(v for v in __version__ if (v.isnumeric() or v == '.'))

with open(f'{__basedir__}/requirements.txt') as f:
    required = f.read().splitlines()
    # e.g. daps-utils @ git+https://github.com/nestauk/daps_utils
    dependency_links = [line.split(' @ ')[-1] for line in required
                        if '://' in line and ' @ ' in line]

exclude = ['docs', 'tests']
common_kwargs = dict(
    version=version,
    license='MIT',
    install_requires=required,
    long_description=open('README.md').read(),
    url='https://github.com/nestauk/ojd_daps',
    author='nesta',
    author_email='software_development@nesta.org.uk',
    maintainer='nesta',
    maintainer_email='software_development@nesta.org.uk',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>3.7',
    include_package_data=True,
    dependency_links=dependency_links,
)

setup(name='ojd_daps',
      packages=find_namespace_packages(where='.', exclude=exclude),
      **common_kwargs)
