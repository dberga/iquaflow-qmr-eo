# Generic
import os
import numpy as np
from typing import Any, Dict, Optional
from glob import glob

# Vision
import cv2

# iquaflow
from iquaflow.datasets import DSModifier


#########################
# Fake Modifier
#########################

class DSModifierFake(DSModifier):
    """
    Class derived from DSModifier that modifies a dataset iterating its folder.
    This modifier copies images from a folder already preexecuted (premodified).

    Args:
        ds_modifer: DSModifier. Composed modifier child

    Attributes:
        name: str. Name of the modifier
        images_dir: str. Directory of images to copy from.
        src_ext : str = 'tif'. Extension of reference GT images
        dst_ext : str = 'tif'. Extension of images to copy from.
        ds_modifer: DSModifier. Composed modifier child
        params: dict. Contains metainfomation of the modifier
        
    """
    def __init__(
        self,
        name: str,
        images_dir: str,
        src_ext : str = 'tif',
        dst_ext : str = 'tif',
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {
            "zoom": 2
        }
    ):
        self.src_ext                = src_ext
        self.dst_ext                = dst_ext
        self.images_dir             = images_dir
        self.name                   = name
        self.params: Dict[str, Any] = params
        self.ds_modifier            = ds_modifier
        self.params.update({"modifier": "{}".format(self.name)})
        
    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        
        os.makedirs(dst, exist_ok=True)
        
        print(f'{self.name} For each image file in <{data_input}>...')
        
        for image_file in glob( os.path.join(data_input,'*.'+self.src_ext) ):
            
            imgp = self._mod_img( image_file )
            dst_file = os.path.join(dst, os.path.basename(image_file))
            if not os.path.exists(dst_file):
                cv2.imwrite( dst_file, imgp )
        
        print('Done.')
        
        return input_name

    def _mod_img(self, image_file: str) -> np.array:
        
        fn = [
            fn for fn in glob(os.path.join(self.images_dir,'*.'+self.dst_ext))
            if os.path.basename(image_file).split('.')[0]==os.path.basename(fn).split('.')[0]
        ][0]
        
        rec_img = cv2.imread(fn)
        
        return rec_img


