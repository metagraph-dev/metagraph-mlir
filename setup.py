from setuptools import setup, find_packages
import versioneer

setup(
    name="metagraph-mlir",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="MLIR plugins for Metagraph",
    author="Anaconda, Inc.",
    packages=find_packages(include=["metagraph_mlir", "metagraph_mlir.*"]),
    include_package_data=True,
    install_requires=["metagraph", "mlir"],
    # entry_points={
    #    "metagraph.plugins": "plugins=metagraph_mlir.registry:find_plugins"
    # },
)
