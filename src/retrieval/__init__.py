"""ArtAgent 检索抽象层。

统一检索入口 HybridRetriever，底下并列多个数据源实现 BaseRetriever：
  - StructuredTableRetriever：结构化表（核心库 + 用户表格）
  - UserDocTextRetriever / UserDocImageRetriever：用户 PDF 两路
  - MuseumAPIRetriever：博物馆开放 API（预留）

每条检索结果带 source 标签保证可追溯性；collection 级隔离，不做物理合并。
"""
