# -*- coding: utf-8 -*-
import requests
import json
import re


triple_relationship = []
relation_set = set()

relation_type = ["爱情", "亲情", "友情"]
keys = ["李宇春"]


def get_json(url):
    # headers = {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'}
    # rsp = requests.get(url, headers=headers)
    rsp = requests.get(url)
    if rsp.status_code == 200:
        return rsp.text
    else:
        print("爬取失败")
        return False


def split_data(data):
    try:
        data_dict = eval(data[:-1])
    except BaseException:
        pass
    else:
        person = {}
        relationship = {}
        # print(type(data_dict))
        nodes = data_dict["nodes"]
        links = data_dict["links"]
        for node in nodes:
            if node["name"] not in keys:
                keys.append(node["name"])
            if node['id'] not in person.keys():
                person[node['id']] = node["name"]
        print(person)
        for link in links:
            # print(link)
            tmp1 = link["from"] + "-" + link["to"]
            tmp2 = link["to"] + "-" + link["from"]
            if tmp1 in relation_set or tmp2 in relation_set:
                continue
            else:
                # relation = {"1": person[link["from"]], "2": person[link["to"]], "r": link["name"]}
                if link["from"] not in person.keys() or link["to"] not in person.keys():
                    continue
                # print(person[link["from"]])
                # print(person[link["to"]])
                # print(link["type"])
                relation = {"1": person[link["from"]], "2": person[link["to"]], "r": relation_type[link["type"]]}
                triple_relationship.append(relation)
                relation_set.add(tmp1)
        print(triple_relationship)


if __name__ == '__main__':
    # base_url = "https://www.sogou.com/tupu/person.html?q="
    try:
        base_url = "https://www.sogou.com/kmap?query=%s&from=relation&id="
        for i in range(0, 50000):
            print(len(triple_relationship))
            key = keys[i]
            target = base_url % key
            print("爬取：", target)
            content = get_json(target)
            if content:
                split_data(content)
    except BaseException as e:
        print('发生错误：',e)
        with open("./data/datasets.json", "a+", encoding="utf-8") as fw:
            fw.write(json.dumps(triple_relationship))
    else:
        with open("./data/datasets.json", "a+", encoding="utf-8") as fw:
            fw.write(json.dumps(triple_relationship))
