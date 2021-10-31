# Sarcasm-style-tranfer
Transferring Styles between Sarcastic and Unsarcastic
关于使用：
由于shap和PPLM使用的transformers版本不同，因此用Colab更好

SHAP部分：要修改存储讽刺数据集和讽刺检测模型的文件路径为自己的
生成部分：需要安装的库在requirements中
因github中上传文件大小的限制，需自行用pplm_discrm_train训练PPLM用的讽刺鉴别器(sarcasm discriminitor)

注：因各个库的版本更新，非之前本地训练好的模型难以跑出，因此方法更值得借鉴

缺陷：因时间和精力问题，作者未实现MLM和RETRIEVE部分(即更好的保留原文，保留结构)

感谢gary老师的指导，感谢PPLM团队的开源模型，感谢huggingface平台的遍历
