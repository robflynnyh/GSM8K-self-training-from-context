from setuptools import setup

setup(
    name='stfc',
    version='0.01',    
    description='',
    url='https://github.com/robflynnyh/GSM8K-self-training-from-context',
    author='Rob Flynn',
    author_email='rjflynn2@sheffield.ac.uk',
    license='Apache 2.0',
    packages=['stfc'],
    install_requires=['torch',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',   
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.8',
    ],
)
