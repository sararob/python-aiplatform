

from setuptools import setup, find_packages


setup(
    name="mlflow_plugin",
    version="0.0.2",
    description="Test plugin for MLflow Tracking with Vertex Experiments",
    packages=find_packages(),
    # Require MLflow as a dependency of the plugin, so that plugin users can simply install
    # the plugin & then immediately use it with MLflow
    install_requires=["mlflow", "google-cloud-aiplatform"],
    entry_points={
        # Define a Tracking Store plugin for tracking URIs with scheme 'file-plugin'
        "mlflow.tracking_store": "vertex-mlflow-plugin=plugin_src.file_store:VertexMlflowTracking",
        # Define a ArtifactRepository plugin for artifact URIs with scheme 'file-plugin'
        # "mlflow.artifact_repository": "file-plugin=mlflow_test_plugin.local_artifact:PluginLocalArtifactRepository",  # pylint: disable=line-too-long
    },
)