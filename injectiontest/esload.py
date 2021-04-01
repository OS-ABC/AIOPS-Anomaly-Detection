from elasticsearch import Elasticsearch
import csv
import sys
import json
import codecs
import importlib

importlib.reload(sys)
# reload(sys)
# sys.setdefaultencoding('utf-8')
# es地址
es = Elasticsearch(["139.9.112.155:9200"], timeout=9999)
# 索引名称和文档类型
es_index = "fault-injection-data"
es_type = "_doc"
# 指定要导出的字段,如果不清楚有哪些字段，但是要导出全部字段，则去掉"_source"部分
csv_header = ["collectTime", "items", "logs","faultInjectionInfo","faultInjectionRet"]
res = es.search(index=es_index, doc_type=es_type, body={
    "query": {
        "match_all": {}
    },
    "_source": {
        "includes": csv_header,
        "excludes": []
    },
}, size=1000)


def export(file_name):
    """
    export es documents to csv file
    :param file_name: 导出数据的目标文件
    :return: None
    """
    mappings = es.indices.get_mapping(index=es_index, doc_type=es_type)
    # export all fields if csv_header is not set
    fields = []
    for field in mappings[es_index]['mappings'][es_type]['properties']:
        fields.append(field)
    if len(csv_header):
        fields = csv_header
    with open(file_name, 'w') as f:
        f.write(codecs.BOM_UTF8)  # 防止整体中文乱码
        header_present = False
        for doc in res['hits']['hits']:
            my_dict = doc['_source']
            if not len(my_dict):
                continue
            if not header_present:
                w = csv.DictWriter(f, fields)
                w.writeheader()
                header_present = True
            deal_chinese_words(my_dict)
            w.writerow(my_dict)


# 对于字典类型数据做特殊处理，转json并把unicode做decode防止乱码。如果文档没有中文那直接注释掉好了
def deal_chinese_words(my_dict):
    if my_dict.get('parameterMap'):
        # in case chinese character garbled by utf8 encoding
        my_dict['parameterMap'] = json.dumps(my_dict['parameterMap']).decode('unicode_escape')
    if my_dict.get('response'):
        my_dict['response'] = json.dumps(my_dict['response']).decode('unicode_escape')


# 执行入口
export("/Users/local/Downloads/data.csv")
