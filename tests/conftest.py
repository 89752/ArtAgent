"""全量测试环境隔离：关闭词法通道的在线翻译，避免测试触发 LLM API。"""

import os

os.environ.setdefault("LEXICAL_TRANSLATE", "0")
os.environ.setdefault("ARTAGENT_JOB_EXECUTION", "inline")
os.environ.setdefault("ARTAGENT_STREAM_EXECUTION", "inline")
