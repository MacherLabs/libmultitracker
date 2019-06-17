from setuptools import setup

setup(name='multitracker',
      version='0.2.1',
      description='Multitracking library. Multiple object tracking using opencv, kalman and dlib trackers.',
      url='https://github.com/MacherLabs/libmultitracker',
      install_requires=[
          'numpy'],
      packages=['multitracker'],
      
      package_data={
        'multitracker':['models/*'],
    },
      zip_safe=False)