import json
from pathlib import Path
import numpy as np
from PIL import Image
from common.io import load_samples
from common.mask import composite_known
from common.metrics import psnr

def test_pair_and_mask(tmp_path: Path):
    warped=np.zeros((8,10,3),dtype=np.uint8); warped[:]=[10,20,30]
    generated=np.zeros_like(warped); generated[:]=[200,210,220]
    mask=np.zeros((8,10),dtype=np.uint8); mask[:,5:]=255
    Image.fromarray(warped).save(tmp_path/"warped_rgb.png"); Image.fromarray(mask).save(tmp_path/"inpaint_mask.png")
    sample=load_samples(str(tmp_path))[0]
    output=np.asarray(composite_known(Image.fromarray(generated),Image.fromarray(warped),Image.fromarray(mask)))
    assert np.array_equal(output[:,:5],warped[:,:5]); assert np.array_equal(output[:,5:],generated[:,5:])
    assert sample.name=="completed_rgb" and psnr(output/255.0,output/255.0)==float("inf")

