# AIOPS-Anomaly-Detection

## 介绍
辅助运维人员进行异常检测，检测数据类型为日志数据和指标数据，内嵌多种异常检测方法，对于使用者来说，可以帮助快速理解和回顾当前的异常检测方法，并容易地重用现有的方法，也可进行进一步的定制或改进，这有助于避免耗时但重复的实验工作。

## KPI异常检测

### Install

```
git clone https://github.com/OS-ABC/AIOPS-Anomaly-Detection.git
cd AIOPS-Anomaly-Detection/kpi
pip install -r requirements.txt
```

#### Dependencies

 `environment.yml` 文件是用 `conda` 管理依赖:

```
conda env create -f environment.yml
```

#### Note

- TensorFlow >= 2.4 is required.

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
- 标签用于评估，在训练中不需要。

#### Sample Script

示例在 `sample/main.py`:

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

## 日志异常检测

### Install

```
git clone https://github.com/OS-ABC/AIOPS-Anomaly-Detection.git
cd AIOPS-Anomaly-Detection/log
pip install -r requirements.txt
```
### Run

#### Log Script

对自己对数据进行采样示例：

` BGL dataset`只包含时间信息，因此适合于时间窗口

#### 1. 构建你自己的日志

-读取原始日志
-提取标签、时间和起源事件
-将事件与模板id匹配
  
*"-" label in bgl represent normal, else label is abnormal.*

`python structure_bgl.py`

#### 2. 滑动窗口或固定窗口进行取样

通过计算不同日志之间的时间差，使用时间窗口进行采样。窗口大小和步长的单位是小时。

If `step_size=0`, it used fixed window; else, it used sliding window

`python sample_bgl.py`


` HDFS dataset`包含块id信息，因此适合按块id分组

*block_id 表示指定的硬盘存储空间*

#### 1. 构建你的日志

和BGL数据集一样的处理

#### 2. 根据 block_id采样

`python sample_hdfs`


Train & Test DeepLog example

```
cd demo
# Train
python deeplog.py train
# Test
python deeplog.py test
```

输出结果、关键参数和训练日志将保存在“result/”路径下


### 构建自己的模型

下面是一个loganomal模型的关键参数示例，它在 `demo/loganomaly.py`  
尝试修改这些参数可以构建自己的模型

```
# Smaple
options['sample'] = "sliding_window"
options['window_size'] = 10

# Features
options['sequentials'] = True
options['quantitatives'] = True
options['semantics'] = False

Model = loganomaly(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
```

## 

