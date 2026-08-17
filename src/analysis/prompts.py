"""门控与分层分析的提示词 + 技法原则参考表。"""

FRAMEWORKS = ("realistic", "abstract", "childlike", "decorative", "not_painting")
FRAMEWORK_LABELS = {
    "realistic": "写实/具象",
    "abstract": "抽象/表现",
    "childlike": "儿童画/涂鸦",
    "decorative": "装饰/图案",
    "not_painting": "非画作",
}

GATE_PROMPT = """你是绘画作品适用框架判定器。请判断这张用户上传的图片属于哪种框架，并输出严格 JSON：
{"framework": "realistic | abstract | childlike | decorative | not_painting",
 "confidence": 0.0-1.0,
 "reason": "判断依据一句话",
 "quality_flags": ["blurry", "tilted", "framed", "partially_cropped", "low_resolution", "too_bright", "too_dark"],
 "content_summary": "画面主体简述"}

判定规则：
- realistic：写实/具象绘画（透视规则适用）；
- abstract：抽象/非具象/表现性绘画（透视不适用）；
- childlike：儿童画/涂鸦/明显低龄手绘（透视降级为空间感知）；
- decorative：装饰画/图案/平面设计类画面（按对称、韵律、配色评价）；
- not_painting：摄影照片、纯文字截图、图表、文档页、UI 截图等非绘画内容。

只输出 JSON，不要解释。"""

PRINCIPLES = """技法原则参考表（每条建议必须引用其中条目）：
- 经典素描/透视原理：一点/两点/三点透视、地平线、消失点、近大远小、空气透视
- 构图：三分法、对称与均衡、黄金分割/螺旋、视觉引导线、留白
- 色彩：伊顿色轮、邻近/互补/三角配色、明度九阶、冷暖对比
- 笔触与材料：厚涂/罩染/干笔、边缘处理、肌理"""

FOCUS_HINTS = {
    "all": "请完整分析三个层次：L1 透视/构图/色彩/线条笔触，L2 风格与情绪基调，L3 专业指导建议。",
    "perspective": "只深入分析透视关系（构图/色彩/笔触简要带过即可），建议只针对透视问题。",
    "composition": "只深入分析构图（透视/色彩/笔触简要带过即可），建议只针对构图问题。",
    "color": "只深入分析色彩关系（透视/构图/笔触简要带过即可），建议只针对色彩问题。",
    "brushwork": "只深入分析线条与笔触（透视/构图/色彩简要带过即可），建议只针对线条笔触问题。",
    "style": "只分析 L2 风格与情绪基调（含流派倾向），不输出 L3 建议。",
}

ANALYSIS_PROMPT = """你是绘画技法与美学分析专家。请分析这张用户上传的画作。

已判定适用框架：{framework_label}（framework={framework}）
本次分析侧重：{focus_hint}

本地度量数据（供参考，与你的视觉判断矛盾时以你看到的画面为准）：
{metrics_json}

质量提示（如有）：{quality_hint}

要求：
1. 严格按 JSON 输出，结构如下：
{{
  "framework": "{framework}",
  "overall_assessment": "一段整体印象（描述性）",
  "layer1_technique": {{
    "perspective": {{"applies": true, "kind": "one_point|two_point|three_point|not_applicable", "vanishing_points": [{{"description": "", "consistency": "unified|inconsistent"}}], "assessment": "", "confidence": 0.0-1.0, "evidence": []}},
    "composition": {{"principles_applied": [], "visual_weight": "", "whitespace": "", "assessment": "", "confidence": 0.0-1.0, "evidence": []}},
    "color": {{"scheme": "", "value_contrast": "强|中|弱", "saturation_tendency": "", "warm_cool": "", "dominant_colors": [], "assessment": "", "confidence": 0.0-1.0, "evidence": []}},
    "line_brushwork": {{"applies": true, "line_quality": "", "brushwork_style": "", "skill_signs": [], "assessment": "", "confidence": 0.0-1.0, "evidence": []}}
  }},
  "layer2_style_mood": {{
    "mood": "",
    "mood_evidence": {{"color": "", "composition": "", "line": ""}},
    "style_affinity": [],
    "caveat": "仅针对画面呈现的视觉特征，不代表作者意图或状态"
  }},
  "layer3_suggestions": {{
    "priority_items": [{{"issue": "", "principle": "", "action": "", "difficulty": "beginner|intermediate|advanced", "location_hint": ""}}]
  }}
}}

2. 只分析画作本身的客观/半客观属性。禁止推断创作者心理状态、人格特质、情绪困扰；禁止心理诊断或心理建议。
3. 情绪氛围（L2 mood）必须同时给出视觉依据（mood_evidence），并保持描述性语气。
4. 建议必须具体可操作：每条含具体位置/数值/改法，且 principle 必须引用下面参考表中的条目（说明是通用美术教学原则）：
{principles}
5. 框架一致性：abstract/childlike/decorative 时 perspective.applies 必须为 false，kind 为 not_applicable；写实/具象时 applies 为 true。
6. 只用 JSON 输出，不要 markdown 代码块，不要多余文字。"""
