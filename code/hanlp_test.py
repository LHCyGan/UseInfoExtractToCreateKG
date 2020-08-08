from pyhanlp import *

sentence = "2016年协鑫集成科技股份有限公司向瑞峰（张家港）光伏科技有限公司支付设备款人民币4，515，770.00元"
res = HanLP.parseDependency(sentence)
print(res)
