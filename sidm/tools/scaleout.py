"""Module to define classes and methods that are helpful for scaleout"""

from dask.distributed import Client, PipInstall


def make_dask_client(address):
    """Create dask client that includes dependency installer"""
    dependencies = [
        "git+https://github.com/SC990987/SIDM.git@SRSTudy",
    ]
    client = Client(address)
    client.register_plugin(PipInstall(packages=dependencies, pip_options=["--upgrade"]))
    return client