# AIOPS-Anomaly-Detection

## 介绍
辅助运维人员进行异常检测，检测数据类型为日志数据和指标数据，内嵌多种异常检测方法，对于使用者来说，可以帮助快速理解和回顾当前的异常检测方法，并容易地重用现有的方法，也可进行进一步的定制或改进，这有助于避免耗时但重复的实验工作。

## 部署

```
docker build dockerfile
```

## KPI异常检测

### 使用运行
#### 一、输入数据说明
输入数据分为训练输入和预测输入，训练输入需要评估模型的准确率等指标。
- 1、模型的训练过程中需要将训练数据放到下面目录中
（data/）
该目录中按照每个csv文件都存储着一系列的数据
数据中每一行的格式为
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

- `timestamp`: 秒级.
- `label`: `0` 正常, `1` 异常
- 标签用于评估，在训练中不需要。

- 2、测试数据输入，需要将数据放入test中，数据的格式为：
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

#### 二、输出数据说明
模型通过timestamp	和value作为输入，再根据这两个数据得到输出label值
训练过程：将通过参数train_rate、valid_rate、test_rate三个值来拆分训练数据、验证数据和测试数据
预测过程：根据输入直接获取到输出打印至控制台

#### 三、模型结构
模型结构简单，通过堆叠多个全连接层作为模型的组织结构
keras.layers.Dense

#### 四、执行命令
训练命令为：
```
cd ~/codes/Anomaly/KPIAnomaly && ~/conda/bin/python3 main.py
```
测试命令为（待测试文件放到test中）：
```
cd /home/nlp/Anomaly/KPIAnomaly && ~/conda/bin/python3 predict.py
```

## 日志异常检测

#### 一、输入数据说明
模型的训练过程中需要将训练数据放到下面目录中
（data/hdfs/）
该目录中按照训练文件名区分，此处将*-slave1.log训练日志，将*-slave2.log作为测试日志
日志中每一行的格式为
<YMD> <Time> <Type> <Component>: <Content>
数据需要先经过deal_log_data.py进行预处理


#### 二、输出数据说明
模型通过输入（经过预处理变成了数字形式）后，经过无监督模型获取到的是当前日志异常值的检测。


#### 三、模型结构
分为两种
模型（LSTM）分四层结构：
0 = {InputLayer} 输入层
1 = {LSTM} lstm连接层 relu激活函数
2 = {LSTM} lstm连接层 relu激活函数
3 = {Dense} 全连接层
矩阵分解（PCA）的方式


#### 四、执行命令

- 采用deeplog方法：
- 1、预处理文件（日志文件存放在data/hdfs中）：
```
cd /home/nlp/Anomaly/LogAnomaly/ && ~/conda3/bin/python3 deal_log_data.py
```

- 2、训练命令为：
```
cd /home/nlp/Anomaly/LogAnomaly/main && ~/conda3/bin/python3 deeplog.py train
```

- 3、测试命令为：
```
cd /home/nlp/Anomaly/LogAnomaly/main && ~/conda3/bin/python3 deeplog.py predict
```
- 采用PCA方法
- 训练命令为：
```
cd /home/nlp/Anomaly/LogAnomaly/pca && ~/conda3/bin/python3 pca_main.py
```
- 采用Robustlog方法
 
- 1、训练命令为：
```
cd /home/nlp/Anomaly/LogAnomaly/main && ~/conda3/bin/python3 robustlog.py train
```

- 2、测试命令为：
```
cd /home/nlp/Anomaly/LogAnomaly/main && ~/conda3/bin/python3 robustlog.py predict
```


