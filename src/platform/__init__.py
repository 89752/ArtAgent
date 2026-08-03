"""ArtAgent 平台层（P0-1 ~ P0-4）。

平台层与领域内核分离：用户/密钥/设置、Agent Manifest 注册表、
OpenAI 兼容 API、MCP 消费侧适配器。领域内核（src/agent、src/retrieval）
不依赖本包；本包只做编排与接入。
"""

PLATFORM_VERSION = "0.1.0"
