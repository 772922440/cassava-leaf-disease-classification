# CV Framework

## Source Tree
```
├── config
│   ├── default.yaml         # 默认配置文件
│   ├── ensemble_eval.yaml   # 集成学习预测配置文件
│   └── ensemble_train.yaml  # 集成学习训练配置文件
├── dataset
│   ├── __init__.py
│   └── leafdisease.py       # 数据集读取和转换
├── debug.py
├── ensemble_eval.py         # 集成学习预测
├── ensemble_split.py        # 集成学习划分 train.csv 为 k 个 folds
├── ensemble_train.py        # 集成学习训练某一个 fold
├── model
│   ├── __init__.py
│   └── resnet.py            # resnet backbone 实现
├── README.md
└── utils
    ├── cls_loss.py          # 分类 Loss 函数
    ├── __init__.py
    ├── torch_utils.py       # PyTorch 辅助函数/scheduler
    └── utils.py             # Python 辅助函数/配置日志系统
```

## 运行命令
* split
  ```
  python ensemble_split.py name=ensemble_train (指定配置文件)
  ```
* train
  ```
  python ensemble_train.py name=ensemble_train k=0
  ```
* eval
```
  python ensemble_eval.py name=ensemble_train
  ```

## 配置系统
* yaml 文件中保存超参数配置
* 配置系统先读取 default.yaml，然后读取 name=xxx.yaml 并且覆盖 default.yaml
* 命令行后面跟随的 key=value也会被读取进去，但默认为字符串格式 （哎 偷懒了）