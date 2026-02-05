# Name: IntegrityCheck
# Model: deepseek-chat
# Temperature: 0.1
# ResponseFormat: json_object

## System Prompt
判断信息是否完整。
输出 JSON: {"is_sufficient": true/false, "suggested_question": "..."}

## User Prompt Template
查询: {rewritten_query}
已知槽位: {current_slots}