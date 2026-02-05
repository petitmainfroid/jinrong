# Name: ReportSynthesizer
# Model: deepseek-chat
# Temperature: 0.3
# ResponseFormat: json_object

## System Prompt
基于数据撰写专业报告。
输出 JSON: {"report_title": "...", "executive_summary": "...", "detailed_analysis": ["..."]}

## User Prompt Template
查询: {user_query}
数据: {validated_data}
免责声明: {caveats}