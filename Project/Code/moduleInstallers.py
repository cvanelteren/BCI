import pip

packages = ['scipy', 'sklearn', 'matplotlib==2.0b4','numpy','seaborn','h5py']

for package in packages:
    pip.main(['install', package])
