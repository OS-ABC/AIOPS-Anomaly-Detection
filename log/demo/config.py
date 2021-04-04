def parse_args():
    options = dict()
    options['data_dir'] = '../data/'  # 训练数据和测试数据的地址目录
    options['window_size'] = 10  # 窗口的大小，主要滑动获取日志信息的 ，数据预处理使用
    options['device'] = "cpu"  # 支持cpu还是gpu设置

    # Smaple
    options['sample'] = "sliding_window"  # 采用滑动窗口的模式

    # Features
    options['sequentials'] = True  # 序列特征
    options['quantitatives'] = True  # 统计特征
    options['semantics'] = False  # 语义特征
    options['feature_num'] = sum(
        [options['sequentials'], options['quantitatives'], options['semantics']])  ##数据集的特征

    # Model
    options['input_size'] = 1  # 输入的大小
    options['hidden_size'] = 64  # 隐藏层的大小
    options['num_layers'] = 2  # 神经网络层数目

    options['num_classes'] = 28  # 分的类别

    # Train
    options['batch_size'] = 2048  # batch_size 批量大小
    options['accumulation_step'] = 1  # 梯度累加的步数

    options['optimizer'] = 'adam'  # 优化器的选择 ‘adam’
    options['lr'] = 0.001   # 学习率
    options['max_epoch'] = 370  # 最大的轮数
    options['lr_step'] = (300, 350) # 学习率的衰减步数
    options['lr_decay_ratio'] = 0.1   # 学习率的衰减率

    options['resume_path'] = None   # 恢复的路径
    options['model_name'] = "deeplog"   # 模型的名字
    options['save_dir'] = "../result/deeplog/"  # 模型的保存的地址

    # Predict
    options['model_path'] = "../result/deeplog/deeplog_last.pth"   # 预训练模型的保存的地址
    options['num_candidates'] = 9  # 候选的数目

    return options


# 输入的主要就是日志的数据信息  比如查看hdfs中的文件
# 测试输出为false positive (FP): 523847, false negative (FN): 294, Precision: 3.061%, Recall: 98.254%, F1-measure: 5.938%
# 包准确率，召回率，精度，错误率, 加权调和平均数

# 训练数据就是data/hdfs中进行修改，参数这个文件夹中修改
# 训练命令 python  deeplog.py train
# 预测命令 python deeplog.py predict

