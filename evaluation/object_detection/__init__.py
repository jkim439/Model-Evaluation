__author__ = 'Junghwan Kim'
__copyright__ = 'Copyright 2016-2019 Junghwan Kim. All Rights Reserved.'
__version__ = '1.0.0'


import numpy as np
import pylab

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


def main(annFile, resFile, annType):


    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)
    imgIds = sorted(cocoGt.getImgIds())
    imgIds = imgIds[0:189]
    imgId = imgIds[np.random.randint(1)]

    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.iouThrs = np.array([0.75, 0.5, 0.1])

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.analyze()
    cocoEval.summarize()
