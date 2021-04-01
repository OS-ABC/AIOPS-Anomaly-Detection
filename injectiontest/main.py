import json
from matplotlib import pyplot as plt
import numpy as np
import time

file_name = "data/fault-injection-data-sample-infi.json"        #文件名
load_f = open(file_name,'r')
load_dict = json.load(load_f)

result = open('result', 'w', encoding='UTF-8')          # 指标结果文件
result_log = open('result_log', 'w', encoding='UTF-8')  # 日志结果文件

#画图函数
def plot_curve1(values, e_values, e_index, title=""):
    x = np.arange(0, len(values), 1)
    plt.figure(figsize=(15,5))
    plt.title(title)
    plt.plot(x, values,'o-')
    plt.plot(e_index, e_values,'*')
    plt.show()

# MAD法: media absolute deviation
def MAD(dataset, n):
    median = np.median(dataset)  # 中位数
    deviations = abs(dataset - median)
    mad = np.median(deviations)
    remove_idx = np.where(abs(dataset - median) > n * mad)[0]

    remove_data = [dataset[i] for i in remove_idx]

    return remove_data, remove_idx

def convertTime(testTime):
    # 转换成localtime
    time_local = time.localtime(testTime)
    # 转换成新的时间格式(2016-05-05 20:28:54)
    testTime = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return testTime

for d in load_dict:
    items = d["_source"]["items"]
    events = d["_source"]["events"]
    _id = d["_id"]
    # assert len(items)==len(events), (len(items),len(events))
    print(_id)

    new_dict = dict()
    new_dict[_id] = []


    for idx, item in enumerate(items):
        item = json.loads(item)
        tempdict = dict()

        allClock = item["allClock"].split(',')
        metricId = item["id"]
        metricName = item["name"]
        belongTo = item["applicationName"]
        hostname = item["hostName"]

        if(item["valueType"] == 0):
            values = item["allValue"].split(',')
            if (len(values) > 4):
                values = [float(v) for v in values]
                e_values = []
                e_index = []
                e_values, e_index = MAD(values[4:], 3)          # 不考虑前4个点
                e_index += 4
                # plot_curve1(values, e_values, e_index)        # 可视化
                if(len(e_index)>0):                             # 输出异常点信息
                    # print("_source-items-%d-allValue:" % (idx))
                    # print(e_index)
                    testTime = [convertTime(int(allClock[m])) for m in e_index]
                    # print("id: %s\n" % _id)
                    # print("testTime: ")
                    # print(testTime)
                    # print("metricId: %s\nmetricName: %s\nbelongTo: %s" % (metricId, metricName, belongTo))
                    # print("value:")
                    # print(e_values)
                    # print("value_index:")
                    # print(e_index)
                    # print("exceptionDegree: Exception")
                    # print()

                    # result.write("id: %s\n" % _id)                      # 输出到result文件
                    # result.write("testTime: ")
                    # result.write(','.join(testTime))
                    # result.write("\n")
                    # result.write("metricId: %s\nmetricName: %s\nbelongTo: %s\n" % (metricId, metricName, belongTo))
                    # result.write("value:")
                    # result.write(','.join([str(v) for v in e_values]))
                    # result.write("\n")
                    # result.write("value_index:")
                    # result.write(','.join([str(e) for e in e_index]))
                    # result.write("\n")
                    # result.write("exceptionDegree: Exception\n")
                    # result.write('\n')
                    tempdict["testTime"] = ','.join(testTime)
                    tempdict["metricId"] = metricId
                    tempdict["metricName"] = metricName
                    tempdict["belongTo"] = belongTo
                    tempdict["value"] = ','.join([str(e_v) for e_v in e_values])
                    tempdict["value_index"] = ','.join([str(e_i) for e_i in e_index])
                    tempdict["hostname"] = hostname
                    tempdict["exceptionDegree"] = "Exception"
                    new_dict[_id].append(tempdict)

    with open("result.json", "w") as f:
        json.dump(new_dict, f)
        # elif(item["valueType"] == 2):
        #     print(item["valueType"])
        #     print(item["allValue"])

    new_dict = dict()
    new_dict[_id] = []

    logs = d["_source"]["logs"]
    logs_keys = [k for k in logs.keys()]
    for idx, log in enumerate(logs):
        # print(logs[log])
        tempdict = dict()

        log = logs[log]
        log_key = logs_keys[idx]

        output = open(log_key, 'w', encoding='UTF-8')           # 日志输出文件初始化
        for i,l in enumerate(log):

            l = json.loads(l)
            if (idx==2 and 'log_message' in l):                 # 日志输出到文件
                # print(i)
                output.write("%i %s\n" % (i, l["log_message"]))

            if('level' in l and l["level"]=="WARN"):            # 打印日志信息

                testTime = l["log_time"]
                log_message = l["log_message"]
                level = l["level"]
                host = l["host"]
                belongTo = host["name"]
                # print("key: %s" % log_key)
                # print("id: %s\n" % _id)
                # print("testTime: %s" % testTime)
                # print("logId: %s:%i" % (log_key, i))
                # print("logDetail: %s" % log_message)
                # print("belongTo: %s:%s" % (belongTo, log_key))
                # # print("exceptionDegree: %s" % level)
                # print()

                # result_log.write("id: %s\n" % _id)
                # result_log.write("testTime: %s\n" % testTime)              # 输出到result文件
                # result_log.write("logId: %s:%i\n" % (log_key, i))
                # result_log.write("logDetail: %s\n" % log_message)
                # result_log.write("belongTo: %s:%s\n" % (belongTo, log_key))
                # result_log.write('\n')

                tempdict["testTime"] = testTime
                tempdict["logId"] = log_key+":"+str(i)
                tempdict["logDetail"] = log_message
                tempdict["belongTo"] = ':'.join([belongTo, log_key])
                new_dict[_id].append(tempdict)

    with open("result_log.json", "w") as f:
        json.dump(new_dict, f)


                # print("_source-logs-%d-level:" % (idx))
                # print(i, l["log_message"])
        output.close()                                      # 关闭日志文件
result.close()
result_log.close()