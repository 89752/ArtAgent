"""ArtAgent 检索抽象层（Stage 2）。

统一检索入口 HybridRetriever，底下并列多个数据源实现 BaseRetriever：
  - StructuredTableRetriever：结构化表（SemArt 首个注册实例 + Stage 5 用户表格）
  - UserDocTextRetriever / UserDocImageRetriever：Stage 3 用户 PDF 两路
  - MuseumAPIRetriever：Stage 7 博物馆开放 API

每条检索结果带 source 标签保证可追溯性；collection 级隔离，不做物理合并。
"""
