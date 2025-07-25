# config/cultural_patterns.py

CULTURAL_PATTERNS = {
    'english': {
        'heading_keywords': [
            'chapter', 'section', 'part', 'introduction', 'conclusion',
            'summary', 'abstract', 'appendix', 'preface', 'table of contents'
        ],
        'numbering_patterns': [
            r'^\d+\.',            # 1.
            r'^\d+\.\d+',         # 1.1
            r'^[IVXLC]+\.',       # I. II. IV.
            r'^[A-Z]\.',          # A. B.
            r'^Figure\s+\d+',     # Figure 1
            r'^Table\s+\d+'       # Table 1
        ],
        'layout_preferences': 'ltr'
    },

    'japanese': {
        'heading_styles': ['章', '節', '項', '第', 'はじめに', '結論'],
        'numbering_patterns': [
            r'^第\d+章',          # 第1章
            r'^\d+\.\d+',         # 1.1
            r'^[一二三四五六七八九十]+、',  # 一、
            r'^（\d+）'           # （1）
        ],
        'layout_preferences': 'vertical_aware'
    },

    'hindi': {
        'heading_styles': ['अध्याय', 'खंड', 'भाग', 'प्रकरण', 'निष्कर्ष', 'परिचय'],
        'numbering_patterns': [
            r'^\d+\.',            # 1.
            r'^[(]?\d+[)]?',      # (1) 1)
            r'^[१२३४५६७८९०]+[.]?', # Hindi numerals with optional dot
            r'^अनुच्छेद\s+\d+'    # अनुच्छेद 1
        ],
        'layout_preferences': 'devanagari_aware'
    },

    'arabic': {
        'heading_styles': ['الفصل', 'القسم', 'الجزء', 'الباب', 'مقدمة', 'خاتمة'],
        'numbering_patterns': [
            r'^الفصل\s+\w+',      # الفصل الأول
            r'^\d+\.',            # 1.
            r'^[\u0660-\u0669]+', # Arabic-Indic numerals ٠١٢٣٤٥٦٧٨٩
            r'^قسم\s+\d+'         # قسم 1
        ],
        'layout_preferences': 'rtl_aware'
    },

    'chinese': {
        'heading_styles': ['章', '节', '部分', '第', '引言', '结论'],
        'numbering_patterns': [
            r'^第[一二三四五六七八九十]+章',   # 第一章
            r'^第\d+章',                   # 第1章
            r'^\d+\.\d+',                  # 1.1
            r'^[一二三四五六七八九十]+、'      # 一、
            r'^（\d+）'                    # （1）
        ],
        'layout_preferences': 'vertical_aware'
    },

    # Optional fallback for other languages you may add later:
    'default': {
        'heading_keywords': ['introduction', 'summary', 'abstract', 'conclusion'],
        'numbering_patterns': [r'^\d+\.', r'^\d+\.\d+'],
        'layout_preferences': 'ltr'
    }
}

TEST_SAMPLES = {
    'english': [
        "Chapter 1: Introduction",
        "1.1 Background",
        "II. Literature Review"
    ],
    'japanese': [
        "第1章 はじめに",
        "1.1 背景",
        "一、概要"
    ],
    'hindi': [
        "अध्याय 1: परिचय",
        "१.१ पृष्ठभूमि",
        "१. प्रस्तावना"
    ],
    'arabic': [
        "الفصل الأول: المقدمة",
        "1.1 الخلفية",
        "١.٢ الهدف"
    ],
    'chinese': [
        "第一章 引言",
        "1.1 背景",
        "一、概述"
    ]
}
