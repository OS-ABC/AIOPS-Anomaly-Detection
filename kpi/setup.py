import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='bagel',
    version='1.7.0',
    license='MIT',
    description='Implementation of Bagel in TensorFlow 2',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/OS-ABC/AIOPS-Anomaly-Detection/tree/main/kpi',
    packages=setuptools.find_packages(include=['bagel', 'bagel.*']),
    platforms='any',
    install_requires=[
        'pandas',
        'scikit-learn',
        'tensorflow',
        'tensorflow-probability',
    ],
    extras_require={
        'dev': [
            'matplotlib',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='~=3.8',
)
