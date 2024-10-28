from setuptools import setup, find_packages

setup(
	name='metas_b_least',
	version='0.1.1',
	author='Michael Wollensack',
	author_email='michael.wollensack@metas.ch',
	description='METAS B LEAST is a Python implementation of the B LEAST program of the ISO 6143:2001 norm',
	long_description=open('README.md').read(),
	long_description_content_type='text/markdown',
	url='https://github.com/wollmich/metas-b-least',
	packages=find_packages(),
	classifiers=[
		'Programming Language :: Python :: 3',
		'License :: OSI Approved :: MIT License',
		'Operating System :: OS Independent',
	],
	install_requires=[
		'numpy',
		'scipy'
	],
	include_package_data=True,
)
