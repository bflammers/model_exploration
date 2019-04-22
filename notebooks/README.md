
## Models under considerations

The following models are discussed more in depth. These models are implemented in the PyOD library, unless stated otherwise.

- [uCBLOF](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.cblof)
- [Isolation Forest](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.iforest)
- [Feature Bagging](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.feature_bagging)
- [Histogram-based Outlier Detection (HBOS)](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.hbos)
- [Outlier Detection with Minimum Covariance Determinant (MCD)](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.mcd)
- [PCA](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.pca)
- [XGBoost Outlier Detector](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.xgbod)

The following models did not make the cut:

- [kNN](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.knn) &rarr; does not learn a model of the data
- [Local Outlier Factor (LOF)](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.lof) &rarr; does not learn a model of the data
- [Angle-based Outlier Detector (ABOD)](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.abod) &rarr; does not learn a model of the data (kNN variant)
- [Local Correlation Integral (LOCI)](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.loci) &rarr; based on code does not seem to learn a model of the data

***
<br/>

#### uCBLOF

#### Isolation Forest

#### Feature Bagging

Takes an ensemble of one type of base detector and adds variance between instances of the base detector by taking random subsets of the features. Default base detector is LOF!! Do not use with LOF because this does not learn a model of the data. Instead try with one of the others under consideration. 

Because the idea behind ensemble methods is to use many weak learners and combine them, it makes sense to pick a simple, fast model as the base detector. Options are HBOS, PCA, 

#### Histogram-based Outlier Detection (HBOS)

Assumes feature independence so takes each feature separately into account. Does not seem like a strong method bu it is fast and simple, so worth a try. 

At first sight seems like it learns a model of the data but check this again more in depth.

#### Outlier Detection with Minimum Covariance Determinant (MCD)

Seems very crude but worth a try. Might be interesting to combine something similar to this with a clustering based method such as uCBLOF - but this is an enhancement for a later stadium. 

#### PCA

This is simple, lightweight and might give good results! Promising method. 

#### XGBoost Outlier Detector

Detector based on XGboost. Not sure exactly how it is used unsupervised. Explanation mentions semi-supervised, so it might need a clean dataset for training. Therefore, even if results are very good, do not immediately give this method a large weight in the final scoring mechanism. 