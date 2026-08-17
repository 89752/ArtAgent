"""报告 schema 常量（Pydantic 模型按需后续演进，V1 用轻量校验）。"""

REQUIRED_TOP = [
    "framework",
    "overall_assessment",
    "layer1_technique",
    "layer2_style_mood",
    "layer3_suggestions",
]

LAYER1_DIMS = ["perspective", "composition", "color", "line_brushwork"]
LAYER1_REQUIRED = ["applies", "assessment", "confidence", "evidence"]
LAYER2_REQUIRED = ["mood", "mood_evidence"]
LAYER3_REQUIRED = ["priority_items"]
