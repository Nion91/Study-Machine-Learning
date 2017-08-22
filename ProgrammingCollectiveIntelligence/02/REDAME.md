# 协同过滤

* 相似度：欧几里得距离/pearson相关系数
* 相似度匹配：找相似的user/item
* user_cf：
  1. 找到相似的users，以相似度作为权重
  2. 根据users对items的rating以及相似度权重计算推荐列表
* item_cf:
  1. 计算所有items之间的相似度矩阵
  2. 以user对items的raing作为权重
  3. 根据权重和相似度矩阵计算推荐列表
