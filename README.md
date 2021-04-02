# AIOPS-Anomaly-Detection

## 介绍
辅助运维人员进行异常检测，检测数据类型为日志数据和指标数据，内嵌多种异常检测方法，对于使用者来说，可以帮助快速理解和回顾当前的异常检测方法，并容易地重用现有的方法，也可进行进一步的定制或改进，这有助于避免耗时但重复的实验工作。

## KPI异常检测

### Install

#### Dependencies

An `environment.yml` is  provided if you prefer `conda` to manage dependencies:

```
conda env create -f environment.yml
```

#### Note

- TensorFlow >= 2.4 is required.
- TensorFlow version is tightly coupled to CUDA and cuDNN so they should be selected carefully.

### Run

#### KPI Format

KPI data must be stored in csv files in the following format:

```
timestamp,   value,       label
1469376000,  0.847300274, 0
1469376300, -0.036137314, 0
1469376600,  0.074292384, 0
1469376900,  0.074292384, 0
1469377200, -0.036137314, 0
1469377500,  0.184722083, 0
1469377800, -0.036137314, 0
1469378100,  0.184722083, 0
```

- `timestamp`: timestamps in seconds (10-digit).
- `label`: `0` for normal points, `1` for anomaly points.
- Labels are used for evaluation and are not required in the production environment.

#### Sample Script

A sample script can be found at `sample/main.py`:

```
cd sample
python main.py
```

### Usage

To prepare the data:

```python
import bagel

kpi = bagel.utils.load_kpi('kpi_data.csv')
kpi.complete_timestamp()
train_kpi, valid_kpi, test_kpi = kpi.split((0.49, 0.21, 0.3))
train_kpi, mean, std = train_kpi.standardize()
valid_kpi, _, _ = valid_kpi.standardize(mean=mean, std=std)
test_kpi, _, _ = test_kpi.standardize(mean=mean, std=std)
```

To construct a Bagel model, train the model, and use the trained model for prediction:

```python
import bagel

model = bagel.Bagel()
model.fit(kpi=train_kpi.use_labels(0.), validation_kpi=valid_kpi, epochs=EPOCHS)
anomaly_scores = model.predict(test_kpi)
```

To save and restore a trained model:

```python
# To save a trained model
model.save(prefix)

# To load a pre-trained model
import bagel

model = bagel.Bagel()
model.load(prefix)
```


