import os
from pystac import (
    Asset,
    Catalog,
    Collection,
    Item,
    Link, LinkType,
    Extent, SpatialExtent, TemporalExtent,
    Extensions, MediaType
)
from pystac.extensions.label import LabelClasses, LabelType
import rasterio
from datetime import datetime, timezone
from shapely.geometry import GeometryCollection, box, shape
from pathlib import Path


class SpaceNetStac(Catalog):
    def __init__(self, id, description, title=None,  href=None):
        super().__init__(id, description, title=title, stac_extensions=[Extensions.LABEL], href=href)

        self.base_file_uri = 'https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/spacenet'
        blank_extent = Extent(SpatialExtent([None, None, None]), TemporalExtent.from_now())

        # Label Collection
        self.label_collection = Collection('label', "Building Labels Collection", blank_extent)
        self.add_child(self.label_collection)
        self.label_href_base = os.path.join(os.path.dirname(self.get_self_href()), 'label')
        self.label_collection.set_self_href(os.path.join(self.label_href_base, 'collection.json'))

        self.label_train_catalog = Catalog('label-train', "Training Split",
            href=os.path.join(self.label_href_base, 'train.json'))
        self.label_collection.add_child(self.label_train_catalog)

        self.label_test_catalog = Catalog('label-test', "Testing Split",
            href=os.path.join(self.label_href_base, 'test.json'))
        self.label_collection.add_child(self.label_test_catalog)

        # Image Collection
        self.image_collection = Collection('image', "Image Chip Collection", blank_extent)
        self.add_child(self.image_collection)
        self.image_href_base = os.path.join(os.path.dirname(self.get_self_href()), 'image')
        self.image_collection.set_self_href(os.path.join(self.image_href_base, 'collection.json'))

        self.image_train_catalog = Catalog('image-train', "Training Split",
            href=os.path.join(self.image_href_base, 'train.json'))
        self.image_collection.add_child(self.image_train_catalog)

        self.image_test_catalog = Catalog('image-test', "Testing Split",
            href=os.path.join(self.image_href_base, 'test.json'))
        self.image_collection.add_child(self.image_test_catalog)


    def spacenet_cell(self, img_file, label_file):
        img_asset_href = os.path.join(self.base_file_uri, img_file)
        label_asset_href = os.path.join(self.base_file_uri, label_file)
        img_id = Path(img_file).stem
        label_id = Path(label_file).stem

        # image and labels share bbox for in spacenet dataset
        bbox = None
        with rasterio.open(img_asset_href) as src:
            bbox = list(src.bounds)

        # STAC Item describes the source image
        image_item = Item(
            id=img_id,
            href=os.path.join(self.image_href_base, img_id+".json"),
            geometry=box(*bbox).__geo_interface__,
            bbox=bbox,
            datetime=datetime.now(timezone.utc),
            properties={},
            stac_extensions=None,
            collection=self.image_collection)

        # STAC Item asset is the actual image chip that was labeled
        image_item.add_asset("rgb", Asset(
            href=img_asset_href,
            title="RGB Chip",
            media_type=MediaType.GEOTIFF))

        # STAC label item describes the labels and classes
        label_item = Item(
            id=label_id,
            href=os.path.join(self.label_href_base, label_id+".json"),
            geometry=box(*bbox).__geo_interface__,
            bbox=bbox,
            properties={},
            datetime=image_item.datetime,
            stac_extensions=[Extensions.LABEL],
            collection=self.label_collection)

        # Apply label extension specific properties
        label_item.ext.label.apply(
            label_type=LabelType.VECTOR,
            label_tasks=['segmentation'],
            label_classes=[LabelClasses(['building', 'background'])],
            label_description='Building Polygons')

        # STAC label item asset is the GeoJSON Feature collection of labels
        label_item.ext.label.add_geojson_labels(
            href=label_asset_href,
            title="Building FeatureCollection")

        # STAC label item links to imagery that was used to trace the labels using "source" the rel
        label_item.ext.label.add_source(image_item)

        self.image_collection.add_item(image_item)
        self.label_collection.add_item(label_item)

        return (image_item, label_item)

    def add_train_set(self, img_file, label_file):
        (img, lbl) = self.spacenet_cell(img_file, label_file)
        self.image_train_catalog.add_item(img)
        self.label_train_catalog.add_item(lbl)

    def add_test_set(self, img_file, label_file):
        (img, lbl) = self.spacenet_cell(img_file, label_file)
        self.image_test_catalog.add_item(img)
        self.label_test_catalog.add_item(lbl)

    def finalize(self):
        """Update collection extends based on the extent of their children"""
        label_bounds = GeometryCollection([shape(s.geometry) for s in self.label_collection.get_all_items()]).bounds
        self.label_collection.extent.spatial = SpatialExtent(label_bounds)

        image_bounds = GeometryCollection([shape(s.geometry) for s in self.image_collection.get_all_items()]).bounds
        self.image_collection.extent.spatial = SpatialExtent(image_bounds)