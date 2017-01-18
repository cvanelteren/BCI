import pip

packages = ['scipy', 'sklearn', 'matplotlib==2.0b4','numpy']

for package in packages:
    pip.main(['install', package])
