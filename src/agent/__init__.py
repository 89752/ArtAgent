"""ArtAgent Agent 内核包。

注意：不要在包 __init__ 里急切 import graph——relevance → agent.prompts →
包初始化 → graph → nodes.general → relevance 会形成循环导入。
"""
