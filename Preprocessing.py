# 乘客的ID、获救情况，乘客等级、姓名、性别、年龄、堂兄弟妹个数、父母与小孩的个数、船票信息、票价、船舱、登船的港口
import pandas as pd
import numpy as np


def preprocesssing(filepath):
    data = pd.read_csv(filepath)

    data["Age"] = data["Age"].where(data["Age"].notnull(), 100)
    data["Embarked"] = data["Embarked"].where(data["Embarked"].notnull(), 0)
    data["Fare"] = data["Fare"].where(data["Fare"].notnull(), 0)

    sex_table = {'female': 1, 'male': 0}
    embarked_table = {'S': 0.25, 'C': 0.5, 'Q': 0.75, 0: 0}
    # 编码
    data['Sex'] = data['Sex'].apply(sex_table.__getitem__)
    data['Embarked'] = data['Embarked'].apply(embarked_table.__getitem__)
    # 删除某些特征
    data.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)
    # 数值类型转换
    data = data.astype('float32')
    # 部分数据归一化
    data["Pclass"] = data["Pclass"] / data["Pclass"].max()
    data["Age"] = data["Age"]/data["Age"].max()
    data["Fare"] = data["Fare"]/data["Fare"].max()
    return data


if __name__ == "__main__":
    train_data = preprocesssing("../DataSet/titanic/train.csv")
    test_data = preprocesssing("../DataSet/titanic/test.csv")

    print(train_data.info())
    print(test_data.info())
