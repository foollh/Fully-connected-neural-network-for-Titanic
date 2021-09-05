# 乘客的ID、获救情况，乘客等级、姓名、性别、年龄、堂兄弟妹个数、父母与小孩的个数、船票信息、票价、船舱、登船的港口
import pandas as pd
import numpy as np


def preprocesssing(filepath):
    data = pd.read_csv(filepath)
    # 删除age和embarked中的空值
    data = data.dropna(subset=['Age', 'Embarked'])

    sex_table = {'female': 1, 'male': 0}
    embarked_table = {'S': 0, 'C': 1, 'Q': 2}
    # 自然数编码
    data['Sex'] = data['Sex'].apply(sex_table.__getitem__)
    data['Embarked'] = data['Embarked'].apply(embarked_table.__getitem__)
    # 删除某些特征
    data.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)
    # 数值类型转换
    data = data.astype('float32')
    # 数据归一化

    return data.values[0:600], data.values[600:712]


if __name__ == "__main__":
    train_data, test_data = preprocesssing("../DataSet/titanic/train.csv")
    # test_data = preprocesssing("../DataSet/titanic/test.csv")

    print(train_data)
    print(test_data)
