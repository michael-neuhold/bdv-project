import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='bdvlib',
    version='0.0.8',
    author='Michael Neuhold',
    author_email='S2110454008@students.fh-hagenberg.at',
    description='...',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/michael-neuhold/bdv-project',
    license='MIT',
    packages=['bdvlib'],
    install_requires=['numpy', 'pandas', 'sklearn', 'scipy', 'matplotlib', 'opencv-python'], # add/replace requirements here
)