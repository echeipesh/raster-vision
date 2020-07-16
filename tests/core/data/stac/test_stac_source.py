import unittest
from rastervision.pipeline import rv_config

from pystac import CatalogType

class TestStacSource(unittest.TestCase):
    def setUp(self):
        self.tmp_dir_obj = rv_config.get_tmp_dir()
        self.tmp_dir = self.tmp_dir_obj.name
        print("\n", self.tmp_dir, "> ", "~" * 100)

    def tearDown(self):
        pass
        # self.tmp_dir_obj.cleanup()

    # TODO: make train/test split catalog
    # TODO: read DatasetConfig from STAC
    # TODO: train RV model form STAC DatasetConfig
    # TODO: parse the Sen1Floods11 catalog
    def test_make_stac(self):
        from . import spacenet_stac
        # I'm going to train on this fucker .. so I better read a catalog

        # TODO: apply best practices from tutorial to this guy
        stac = spacenet_stac.SpaceNetStac('tiny-spacenet', 'Tiny SpaceNet Subset', href='/opt/data/tmp/catalog.json')
        stac.add_training_set(
            img_file='RGB-PanSharpen_AOI_2_Vegas_img205.tif',
            label_file='buildings_AOI_2_Vegas_img205.geojson')

        stac.finalize()
        stac.describe()
        stac.save(catalog_type=CatalogType.SELF_CONTAINED)

        raise Exception("Half baked")


if __name__ == '__main__':
    unittest.main()
