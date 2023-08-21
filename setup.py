from setuptools import setup, Extension

def get_version_number():
    main_ns = {}
    for line in open('pyqchem/__init__.py', 'r').readlines():
        if not(line.find('__version__')):
            exec(line, main_ns)
            return main_ns['__version__']


# Make python package
setup(name='pyqchem',
      version=get_version_number(),
      description='Python wrapper for Q-Chem',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      install_requires=['numpy', 'scipy', 'lxml', 'requests', 'matplotlib', 'PyYAML'],
      author='Abel Carreras',
      author_email='abelcarreras83@gmail.com',
      ext_modules=[
        Extension(
            'pyqchem.functions_cython.efficient_functions',
            sources=['pyqchem/functions_cython/efficient_functions.pyx'],
        )],
      packages=['pyqchem', 'pyqchem.parsers', 'pyqchem.parsers.common','pyqchem.tools'],
      url='https://github.com/abelcarreras/PyQchem',
      classifiers=[
          "Programming Language :: Python",
          "License :: OSI Approved :: MIT License"]
      )
