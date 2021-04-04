# AIOPS-Anomaly-Detection

## 介绍
辅助运维人员进行异常检测，检测数据类型为日志数据和指标数据，内嵌多种异常检测方法，对于使用者来说，可以帮助快速理解和回顾当前的异常检测方法，并容易地重用现有的方法，也可进行进一步的定制或改进，这有助于避免耗时但重复的实验工作。

## KPI异常检测

### Install

通常，安装此软件包时，`pip`将自动安装所需的PyPI依赖项:

- 供开发使用:

  ```
  git clone https://github.com/OS-ABC/AIOPS-Anomaly-Detection.git
  cd AIOPS-Anomaly-Detection/kpi
  pip install -e .[dev]
  ```

- 供生产使用:

  ```
  pip install git+https://github.com/OS-ABC/AIOPS-Anomaly-Detection.git
  ```


#### Dependencies


 `environment.yml` 文件是用 `conda` 管理依赖:

```
conda env create -f environment.yml
```
### Note

- 注意TensorFlow >= 2.4

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

#### Usage

准备数据:

```python
import bagel

kpi = bagel.utils.load_kpi('kpi_data.csv')
kpi.complete_timestamp()
train_kpi, valid_kpi, test_kpi = kpi.split((0.49, 0.21, 0.3))
train_kpi, mean, std = train_kpi.standardize()
valid_kpi, _, _ = valid_kpi.standardize(mean=mean, std=std)
test_kpi, _, _ = test_kpi.standardize(mean=mean, std=std)
```

构建模型，对模型进行训练，并利用训练后的模型进行预测:

```python
import bagel

model = bagel.Bagel()
model.fit(kpi=train_kpi.use_labels(0.), validation_kpi=valid_kpi, epochs=epochs)
anomaly_scores = model.predict(test_kpi)
```

保存和恢复经过训练的模型:

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

#### Note

- 使用torch==1.6.0+的版本 否则 predict会报错

### Run

#### Log Script

自己对数据进行采样示例：

#### BGL数据集
` BGL dataset`只包含时间信息，因此适合于时间窗口

##### 1. 构建你自己的日志

-读取原始日志
-提取标签、时间和起源事件
-将事件与模板id匹配
  
*"-" label in bgl represent normal, else label is abnormal.*

`python structure_bgl.py`

##### 2. 滑动窗口或固定窗口进行取样

通过计算不同日志之间的时间差，使用时间窗口进行采样。窗口大小和步长的单位是小时。

If `step_size=0`, it used fixed window; else, it used sliding window

`python sample_bgl.py`

#### HDFS数据集
` HDFS dataset`包含块id信息，因此适合按块id分组

*block_id 表示指定的硬盘存储空间*

##### 1. 构建你自己的日志

和BGL数据集一样的处理

##### 2. 根据 block_id采样

`python sample_hdfs`

#### 使用样例

数据准备好进行训练和测试，可在confg里修改参数，选择异常检测方法，例如使用 DeepLog方法去检测

```
cd demo
# Train
python deeplog.py train
# Test
python deeplog.py test
```

输出结果、关键参数和训练日志将保存在“result/”路径下


### 构建自己的模型


可以在demo/config尝试修改参数构建自己的模型

下面是一个loganomal模型的关键参数示例，它在 `demo/loganomaly.py`  

```
# Smaple
options['sample'] = "sliding_window" # 滑动窗口
options['window_size'] = 10 # 窗口的大小，主要滑动获取日志信息的

# Features
options['sequentials'] = True # 序列特征
options['quantitatives'] = True # 统计特征
options['semantics'] = False # 语义特征

Model = loganomaly(input_size=options['input_size'], # 输入的大小
                    hidden_size=options['hidden_size'],# 隐藏层的大小
                    num_layers=options['num_layers'],# 神经网络层数目
                    num_keys=options['num_classes']) # 分的类别
                     
```
下面是一个deeplog模型的关键参数示例，它在 `demo/loganomaly.py`  

```
# Smaple
options['sample'] = "sliding_window"
options['window_size'] = 10  

# Features
options['sequentials'] = True
options['quantitatives'] = False
options['semantics'] = False

Model = deeplog(input_size=options['input_size'], # 输入的大小
                    hidden_size=options['hidden_size'],# 隐藏层的大小
                    num_layers=options['num_layers'],# 神经网络层数目
                    num_keys=options['num_classes']) # 分的类别
                     
```
下面是一个robustlog模型的关键参数示例，它在 `demo/loganomaly.py`  

```
# Smaple
options['sample'] = "session_window" # 会话窗口
options['window_size'] = -1

# Features
options['sequentials'] = False
options['quantitatives'] = False
options['semantics'] = True

Model = robustlog(input_size=options['input_size'], # 输入的大小
                    hidden_size=options['hidden_size'],# 隐藏层的大小
                    num_layers=options['num_layers'],# 神经网络层数目
                    num_keys=options['num_classes']) # 分的类别
                     
```

## 

