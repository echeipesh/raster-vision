import unittest

from pystac import (
    CatalogType
)
from pystac.extensions.label import LabelClasses, LabelType


class TestStacMake(unittest.TestCase):
    def setUp(self):
        print("\n", "-" * 70)

    def test_make_stac(self):
        # TODO record label classes in the MlStac collection

        from . import spacenet_stac
        stac = spacenet_stac.SpaceNetStac(
            'tiny-spacenet', 'Tiny SpaceNet Subset',
            href='/opt/src/stac/extensions/ml-data/examples/spacenet/catalog.json')

        from . import ml_stac
        tcal = ml_stac.MlStac(
            'tiny-spacenet-split', 'ML Data training catalog from SpaceNet',
            href='/opt/src/stac/extensions/ml-data/examples/ml-data/catalog.json',
            label_classes=LabelClasses(['building', 'background']),
            label_type=LabelType.VECTOR,
            label_tasks='segmentation')

        (i1, l1) = stac.add_item_pair(
            img_file='RGB-PanSharpen_AOI_2_Vegas_img205.tif',
            label_file='buildings_AOI_2_Vegas_img205.geojson')

        (i2, l2) = stac.add_item_pair(
            img_file='RGB-PanSharpen_AOI_2_Vegas_img25.tif',
            label_file='buildings_AOI_2_Vegas_img25.geojson')

        tcal.add_train_pair(i1, l1)
        tcal.add_test_pair(i2, l2)

        stac.finalize()

        stac.describe()
        tcal.describe()

        stac.save(catalog_type=CatalogType.SELF_CONTAINED)
        tcal.save(catalog_type=CatalogType.SELF_CONTAINED)


if __name__ == '__main__':
    unittest.main()
