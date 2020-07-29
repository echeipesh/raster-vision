# flake8: noqa

from os.path import join

from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *

from pystac import (
    Catalog,
    Collection,
    Item)
from rastervision.core.data import (
    ClassConfig,
    RasterioSourceConfig,
    StatsTransformerConfig,
    SemanticSegmentationLabelSourceConfig,
    RasterizedSourceConfig,
    GeoJSONVectorSourceConfig,
    RasterizerConfig,
    SceneConfig,
    DatasetConfig)
from typing import Generator

import colored_traceback
colored_traceback.add_hook()

def make_scene(item: Item, channel_order: [int]) -> SceneConfig:
    # TODO what is asset name is different?
    image_uri = item.assets['rgb'].href
    label_uri = item.assets['labels'].href

    raster_source = RasterioSourceConfig(
        uris=[image_uri],
        channel_order=channel_order,
        transformers=[StatsTransformerConfig()])

    # TODO: support raster labels for semantic segmentation
    vector_source = GeoJSONVectorSourceConfig(
        uri=label_uri, default_class_id=0, ignore_crs_field=True)

    label_source = SemanticSegmentationLabelSourceConfig(
        raster_source=RasterizedSourceConfig(
            vector_source=vector_source,
            rasterizer_config=RasterizerConfig(background_class_id=1)))

    return SceneConfig(
        id=item.id,
        raster_source=raster_source,
        label_source=label_source)


def make_scenes(collection: Collection, channel_order: [int]) -> Generator[SceneConfig, None, None]:
    return [make_scene(x, channel_order) for x in collection.get_all_items()]


def read_config(catalog: Catalog) -> DatasetConfig:
    # TODO: read ClassConfig from collection  source.properties['label:classes']
    class_config = ClassConfig(names=['building', 'background'], colors=['red', 'black'])
    channel_order = [0, 1, 2]

    # Read taining scenes from STAC
    train_collection = catalog.get_child(id='train')
    train_scenes = make_scenes(train_collection, channel_order)

    # Read testing scenes from STAC
    test_collection = catalog.get_child(id='test')
    test_scenes = make_scenes(test_collection, channel_order)

    return DatasetConfig(
        class_config=class_config,
        train_scenes=train_scenes,
        validation_scenes=test_scenes)

def get_config(runner):
    root_uri = '/opt/data/output/'
    data: Catalog = Catalog.from_file("/opt/src/stac/extensions/ml-data/examples/ml-data/catalog.json")
    data.describe()

    dataset = read_config(data)

    # Use the PyTorch backend for the SemanticSegmentation pipeline.
    chip_sz = 300
    backend = PyTorchSemanticSegmentationConfig(
        model=SemanticSegmentationModelConfig(backbone=Backbone.resnet50),
        solver=SolverConfig(lr=1e-4, num_epochs=1, batch_sz=2))
    chip_options = SemanticSegmentationChipOptions(
        window_method=SemanticSegmentationWindowMethod.random_sample,
        chips_per_scene=10)

    return SemanticSegmentationConfig(
        root_uri=root_uri,
        dataset=dataset,
        backend=backend,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz,
        chip_options=chip_options)
