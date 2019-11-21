# GEmodel-CIKM16--
本仓库是对CIKM16文章(Learning Graph-based POI Embedding for Location-based Recommendation)的复现.

---

### 使用办法

1. 下载checkin数据(https://www.dropbox.com/sh/uhye7orwjz1fn9l/AADDhBogROVDA9jsivguIguKa?dl=0)
2. 运行model_in_gen.py，其中：
   - train_frac设置训练集占比
   - delta_t设置时间间隔，只需修改第一个整数即可
3. 运行GEmodel c++程序，生成嵌入向量
4. 运行topNscore.py，其中：
   - N设置top**N**
   - sample_num设置本次测试样本数量
   - delta_time_weight设置用户向量生成权重
   - 在该代码中，应当先运行dicts_gen()函数以产生字典(不需要再运行get_dicts())，第二次运行程序时可以不调用dicts_gen()。user_vec_gen()受限于sample_num，每次并不一致，如果需要重新抽样，请重新运行该函数，否则可以使用get_user_vec()读取用户向量

PS: 原数据集中，有一行的POI名与众不同，由6个(?)数字组成，可能会造成影响，请注意。

---

仓库中cpp程序来自Dr. Hongzhi Yin主页(https://sites.google.com/site/dbhongzhi/ )，我只修改了部分bug。论文引用为：

> Xie M, Yin H, Wang H, et al. Learning graph-based poi embedding for location-based recommendation[C]//Proceedings of the 25th ACM International on Conference on Information and Knowledge Management. ACM, 2016: 15-24.