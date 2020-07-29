import os
from pystac import (
    Catalog,
    Collection,
    Item,
    Link,
    Extent, SpatialExtent, TemporalExtent,
    MediaType
)
from pystac.extensions.label import LabelClasses, LabelType
from datetime import datetime, timezone
from dataclasses import dataclass


def make_rv_stac_item(image_href_base, image_item: Item, label_item: Item):
    item_id = image_item.id

    item = Item(
        id=item_id,
        href=os.path.join(image_href_base, "{}.json".format(item_id)),
        geometry=image_item.geometry,
        bbox=image_item.bbox,
        datetime=datetime.now(timezone.utc),
        properties={},
        stac_extensions=None,
        collection=None)  # TODO: we have no collection here?

    item.add_asset('rgb', image_item.assets['rgb'])
    item.add_asset('labels', label_item.assets['labels'])
    item.add_links([
        Link(rel='derived_from', target=image_item, media_type=MediaType.JSON),
        Link(rel='derived_from', target=label_item, media_type=MediaType.JSON)
    ])

    return item


class MlStac(Catalog):
    """Catalog of ML sample items with training/testing split"""
    def __init__(self, id, description, title=None, href=None,
                 label_type=None, label_tasks=None, label_classes=None):

        super().__init__(id, description, title=title, stac_extensions=[], href=href)

        label_props = {}
        if label_type is not None:
            label_props['label:type'] = label_type
        if label_tasks is not None:
            label_props['label:tasks'] = label_tasks
        if label_classes is not None:
            label_props['label:classes'] = label_classes.to_dict()

        self.train_catalog = Collection(
            id='train',
            description="Training Split",
            extent=Extent(SpatialExtent([None, None, None]), TemporalExtent.from_now()),
            properties=label_props)
        self.train_catalog_dir = os.path.join(os.path.dirname(self.get_self_href()), 'train')
        self.train_catalog.set_self_href(os.path.join(self.train_catalog_dir, 'collection.json'))
        self.add_child(self.train_catalog)

        self.test_catalog = Collection(
            id='test',
            description="Testing Split",
            extent=Extent(SpatialExtent([None, None, None]), TemporalExtent.from_now()),
            properties=label_props)
        self.test_catalog_dir = os.path.join(os.path.dirname(self.get_self_href()), 'test')
        self.test_catalog.set_self_href(os.path.join(self.test_catalog_dir, 'collection.json'))
        self.add_child(self.test_catalog)

    def add_train_pair(self, image_item: Item, label_item: Item):
        item = make_rv_stac_item(self.train_catalog_dir, image_item, label_item)
        self.train_catalog.add_item(item)
        return item

    def add_test_pair(self, image_item: Item, label_item: Item):
        item = make_rv_stac_item(self.test_catalog_dir, image_item, label_item)
        self.test_catalog.add_item(item)
        return item
