import sys
sys.path.append('/home/jkim')

# Classification
print('\n--------------------\nClassification\n--------------------')
import evaluation.classification as ec
ec.scoring(y_true, y_pred)


# Object Detection
print('\n--------------------\nObject Detection\n--------------------')
import evaluation.object_detection as eo
eo.main('/home/jkim/nas/public_data/MS_COCOdataset/annotations/instances_val2014.json', '/home/jkim/nas/public_data/MS_COCOdataset/results/instances_val2014_fakebbox100_results.json', 'bbox')


# Regression
print('\n--------------------\nRegression\n--------------------')
import evaluation.regression as er
er.evs(y_true, y_pred)
er.max(y_true, y_pred)
er.mae(y_true, y_pred)
er.mse(y_true, y_pred)
er.msle(y_true, y_pred)
er.mdae(y_true, y_pred)
er.r2(y_true, y_pred)


# Segmentation
print('\n--------------------\nSegmentation\n--------------------')
import evaluation.segmentation as es
sys.path.append('/home/jkim/evaluation/segmentation')
es.main()
