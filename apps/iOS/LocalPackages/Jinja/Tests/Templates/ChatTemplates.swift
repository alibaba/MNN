//
//  ChatTemplates.swift
//  Jinja
//
//  Created by Anthony DePasquale on 02.01.2025.
//

struct ChatTemplate {
    static let llama3_1 = """
        {{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n
        """
    static let llama3_2 = """
        {{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now(\"%d %b %Y\") %}\n    {%- else %}\n        {%- set date_string = \"26 Jul 2024\" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n        {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n        {{- '\"parameters\": ' }}\n        {{- tool_call.arguments | tojson }}\n        {{- \"}\" }}\n        {{- \"<|eot_id|>\" }}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n
        """
    static let qwen2_5 = """
        {%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n
        """
    static let mistral7b = """
        {{bos_token}}{% set user_messages = messages | selectattr('role', 'equalto', 'user') | list %}{% for message in messages %}{% if message['role'] == 'user' %}{% if message == user_messages[-1] %}{% if tools %}{{'[AVAILABLE_TOOLS]'+ tools|string + '[/AVAILABLE_TOOLS]'}}{% endif %}{{ '[INST]' + message['content'] + '[/INST]' }}{% else %}{{ '[INST]' + message['content'] + '[/INST]' }}{% endif %}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' ' + eos_token}}{% elif message['role'] == 'tool_results' %}{{'[TOOL_RESULTS]' + message['content']|string + '[/TOOL_RESULTS]'}}{% elif message['role'] == 'tool_calls' %}{{'[TOOL_CALLS]' + message['content']|string + eos_token}}{% endif %}{% endfor %}
        """

    static let granite3_3 = """
        {# Alias tools -> available_tools #}\n{%- if tools and not available_tools -%}\n    {%- set available_tools = tools -%}\n{%- endif -%}\n{%- if messages[0]['role'] == 'system' %}\n     {%- set system_message = messages[0]['content'] %}\n     {%- set loop_messages = messages[1:] %}\n {%- else %}\n     {%- set system_message = \"Knowledge Cutoff Date: April 2024.\\nToday's Date: \" + strftime_now('%B %d, %Y') + \".\\nYou are Granite, developed by IBM.\" %}\n     {%- if available_tools and documents %}\n         {%- set system_message = system_message + \" You are a helpful assistant with access to the following tools. When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.\\nWrite the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data.\" %}\n     {%- elif available_tools %}\n         {%- set system_message = system_message + \" You are a helpful assistant with access to the following tools. When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.\" %}\n     {%- elif documents %}\n         {%- set system_message = system_message + \" Write the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data.\" %}\n    {%- elif thinking %}\n    {%- set system_message = system_message + \" You are a helpful AI assistant.\\nRespond to every user query in a comprehensive and detailed way. You can write down your thoughts and reasoning process before responding. In the thought process, engage in a comprehensive cycle of analysis, summarization, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. In the response section, based on various attempts, explorations, and reflections from the thoughts section, systematically present the final solution that you deem correct. The response should summarize the thought process. Write your thoughts between <think></think> and write your response between <response></response> for each user query.\" %}\n     {%- else %}\n         {%- set system_message = system_message + \" You are a helpful AI assistant.\" %}\n     {%- endif %}\n     {%- if 'citations' in controls and documents %}\n         {%- set system_message = system_message + '\\nUse the symbols <|start_of_cite|> and <|end_of_cite|> to indicate when a fact comes from a document in the search result, e.g <|start_of_cite|> {document_id: 1}my fact <|end_of_cite|> for a fact from document 1. Afterwards, list all the citations with their corresponding documents in an ordered list.' %}\n     {%- endif %}\n     {%- if 'hallucinations' in controls and documents %}\n         {%- set system_message = system_message + '\\nFinally, after the response is written, include a numbered list of sentences from the response with a corresponding risk value that are hallucinated and not based in the documents.' %}\n     {%- endif %}\n     {%- set loop_messages = messages %}\n {%- endif %}\n {{- '<|start_of_role|>system<|end_of_role|>' + system_message + '<|end_of_text|>\\n' }}\n {%- if available_tools %}\n     {{- '<|start_of_role|>available_tools<|end_of_role|>' }}\n     {{- available_tools | tojson(indent=4) }}\n     {{- '<|end_of_text|>\\n' }}\n {%- endif %}\n {%- if documents %}\n     {%- for document in documents %}\n         {{- '<|start_of_role|>document {\"document_id\": \"' + document['doc_id'] | string + '\"}<|end_of_role|>\\n' }}\n         {{- document['text'] }}\n         {{- '<|end_of_text|>\\n' }}\n              {%- endfor %}\n {%- endif %}\n {%- for message in loop_messages %}\n     {{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' + message['content'] + '<|end_of_text|>\\n' }}\n     {%- if loop.last and add_generation_prompt %}\n         {{- '<|start_of_role|>assistant' }}\n             {%- if controls %}\n                 {{- ' ' + controls | tojson()}}\n             {%- endif %}\n         {{- '<|end_of_role|>' }}\n     {%- endif %}\n {%- endfor %}
        """
    static let smollm3 = #"""
        {# ───── defaults ───── #}
        {%- if enable_thinking is not defined -%}
        {%- set enable_thinking = true -%}
        {%- endif -%}

        {# ───── reasoning mode ───── #}
        {%- if enable_thinking -%}
        {%- set reasoning_mode = "/think" -%}
        {%- else -%}
        {%- set reasoning_mode = "/no_think" -%}
        {%- endif -%}

        {# ───── header (system message) ───── #}
        {{- "<|im_start|>system\n" -}}

        {%- if messages[0].role == "system" -%}
        {%- set system_message = messages[0].content -%}
        {%- if "/no_think" in system_message -%}
        {%- set reasoning_mode = "/no_think" -%}
        {%- elif "/think" in system_message -%}
        {%- set reasoning_mode = "/think" -%}
        {%- endif -%}
        {%- set custom_instructions = system_message.replace("/no_think", "").replace("/think", "").rstrip() -%}
        {%- endif -%}

        {%- if "/system_override" in system_message -%}
        {{- custom_instructions.replace("/system_override", "").rstrip() -}}
        {{- "<|im_end|>\n" -}}
        {%- else -%}
        {{- "## Metadata\n\n" -}}
        {{- "Knowledge Cutoff Date: June 2025\n" -}}
        {%- set today = strftime_now("%d %B %Y") -%}
        {{- "Today Date: " ~ today ~ "\n" -}}
        {{- "Reasoning Mode: " + reasoning_mode + "\n\n" -}}

        {{- "## Custom Instructions\n\n" -}}
        {%- if custom_instructions -%}
        {{- custom_instructions + "\n\n" -}}
        {%- elif reasoning_mode == "/think" -%}
        {{- "You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.\n\n" -}}
        {%- else -%}
        {{- "You are a helpful AI assistant named SmolLM, trained by Hugging Face.\n\n" -}}
        {%- endif -%}

        {%- if xml_tools or python_tools -%}
        {{- "### Tools\n\n" -}}
        {%- if xml_tools -%}
          {%- set ns = namespace(xml_tool_string="You may call one or more functions to assist with the user query.\nYou are provided with function signatures within <tools></tools> XML tags:\n\n<tools>\n") -%}
          {%- for tool in xml_tools[:] -%} {# The slicing makes sure that xml_tools is a list #}
            {%- set ns.xml_tool_string = ns.xml_tool_string ~ (tool | string) ~ "\n" -%}
          {%- endfor -%}
          {%- set xml_tool_string = ns.xml_tool_string + "</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>" -%}
          {{- xml_tool_string -}}
        {%- endif -%}
        {%- if python_tools -%}
          {%- set ns = namespace(python_tool_string="When you send a message containing Python code between '<code>' and '</code>' tags, it will be executed in a stateful Jupyter notebook environment, and you will then be given the output to continued reasoning in an agentic loop.\n\nYou can use the following tools in your python code like regular functions:\n<tools>\n") -%}
          {%- for tool in python_tools[:] -%} {# The slicing makes sure that python_tools is a list #}
            {%- set ns.python_tool_string = ns.python_tool_string ~ (tool | string) ~ "\n" -%}
          {%- endfor -%}
          {%- set python_tool_string = ns.python_tool_string + "</tools>\n\nThe state persists between code executions: so variables that you define in one step are still available thereafter." -%}
          {{- python_tool_string -}}
        {%- endif -%}
        {{- "\n\n" -}}
        {{- "<|im_end|>\n" -}}
        {%- endif -%}
        {%- endif -%}
        {# ───── main loop ───── #}
        {%- for message in messages -%}
        {%- set content = message.content if message.content is string else "" -%}
        {%- if message.role == "user" -%}
            {{ "<|im_start|>" + message.role + "\n"  + content + "<|im_end|>\n" }}
        {%- elif message.role == "assistant" -%}
            {% generation %}
            {%- if reasoning_mode == "/think" -%}
                {{ "<|im_start|>assistant\n" + content.lstrip("\n") + "<|im_end|>\n" }}
            {%- else -%}
                {{ "<|im_start|>assistant\n" + "<think>\n\n</think>\n" + content.lstrip("\n") + "<|im_end|>\n" }}
            {%- endif -%}
            {% endgeneration %}
        {%- elif message.role == "tool" -%}
        {{ "<|im_start|>" + "user\n"  + content + "<|im_end|>\n" }}
        {%- endif -%}
        {%- endfor -%}
        {# ───── generation prompt ───── #}
        {%- if add_generation_prompt -%}
        {%- if reasoning_mode == "/think" -%}
            {{ "<|im_start|>assistant\n" }}
        {%- else -%}
            {{ "<|im_start|>assistant\n" + "<think>\n\n</think>\n"  }}
        {%- endif -%}
        {%- endif -%}
        """#

    static let qwen3_coder = #"""
        {% macro render_item_list(item_list, tag_name='required') %}
            {%- if item_list is defined and item_list is iterable and item_list | length > 0 %}
                {%- if tag_name %}{{- '\n<' ~ tag_name ~ '>' -}}{% endif %}
                    {{- '[' }}
                        {%- for item in item_list -%}
                            {%- if loop.index > 1 %}{{- ", "}}{% endif -%}
                            {%- if item is string -%}
                                {{ "`" ~ item ~ "`" }}
                            {%- else -%}
                                {{ item }}
                            {%- endif -%}
                        {%- endfor -%}
                    {{- ']' }}
                {%- if tag_name %}{{- '</' ~ tag_name ~ '>' -}}{% endif %}
            {%- endif %}
        {% endmacro %}

        {%- if messages[0]["role"] == "system" %}
            {%- set system_message = messages[0]["content"] %}
            {%- set loop_messages = messages[1:] %}
        {%- else %}
            {%- set loop_messages = messages %}
        {%- endif %}

        {%- if not tools is defined %}
            {%- set tools = [] %}
        {%- endif %}

        {%- if system_message is defined %}
            {{- "<|im_start|>system\n" + system_message }}
        {%- else %}
            {%- if tools is iterable and tools | length > 0 %}
                {{- "<|im_start|>system\nYou are Qwen, a helpful AI assistant that can interact with a computer to solve tasks." }}
            {%- endif %}
        {%- endif %}
        {%- if tools is iterable and tools | length > 0 %}
            {{- "\n\nYou have access to the following functions:\n\n" }}
            {{- "<tools>" }}
            {%- for tool in tools %}
                {%- if tool.function is defined %}
                    {%- set tool = tool.function %}
                {%- endif %}
                {{- "\n<function>\n<name>" ~ tool.name ~ "</name>" }}
                {{- '\n<description>' ~ (tool.description | trim) ~ '</description>' }}
                {{- '\n<parameters>' }}
                {%- for param_name, param_fields in tool.parameters.properties|items %}
                    {{- '\n<parameter>' }}
                    {{- '\n<name>' ~ param_name ~ '</name>' }}
                    {%- if param_fields.type is defined %}
                        {{- '\n<type>' ~ (param_fields.type | string) ~ '</type>' }}
                    {%- endif %}
                    {%- if param_fields.description is defined %}
                        {{- '\n<description>' ~ (param_fields.description | trim) ~ '</description>' }}
                    {%- endif %}
                    {{- render_item_list(param_fields.enum, 'enum') }}
                    {%- set handled_keys = ['type', 'description', 'enum', 'required'] %}
                    {%- for json_key in param_fields.keys() | reject("in", handled_keys) %}
                        {%- set normed_json_key = json_key | replace("-", "_") | replace(" ", "_") | replace("$", "") %}
                        {%- if param_fields[json_key] is mapping %}
                            {{- '\n<' ~ normed_json_key ~ '>' ~ (param_fields[json_key] | tojson | safe) ~ '</' ~ normed_json_key ~ '>' }}
                        {%- else %}
                            {{-'\n<' ~ normed_json_key ~ '>' ~ (param_fields[json_key] | string) ~ '</' ~ normed_json_key ~ '>' }}
                        {%- endif %}
                    {%- endfor %}
                    {{- render_item_list(param_fields.required, 'required') }}
                    {{- '\n</parameter>' }}
                {%- endfor %}
                {{- render_item_list(tool.parameters.required, 'required') }}
                {{- '\n</parameters>' }}
                {%- if tool.return is defined %}
                    {%- if tool.return is mapping %}
                        {{- '\n<return>' ~ (tool.return | tojson | safe) ~ '</return>' }}
                    {%- else %}
                        {{- '\n<return>' ~ (tool.return | string) ~ '</return>' }}
                    {%- endif %}
                {%- endif %}
                {{- '\n</function>' }}
            {%- endfor %}
            {{- "\n</tools>" }}
            {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
        {%- endif %}
        {%- if system_message is defined %}
            {{- '<|im_end|>\n' }}
        {%- else %}
            {%- if tools is iterable and tools | length > 0 %}
                {{- '<|im_end|>\n' }}
            {%- endif %}
        {%- endif %}
        {%- for message in loop_messages %}
            {%- if message.role == "assistant" and message.tool_calls is defined and message.tool_calls is iterable and message.tool_calls | length > 0 %}
                {{- '<|im_start|>' + message.role }}
                {%- if message.content is defined and message.content is string and message.content | trim | length > 0 %}
                    {{- '\n' + message.content | trim + '\n' }}
                {%- endif %}
                {%- for tool_call in message.tool_calls %}
                    {%- if tool_call.function is defined %}
                        {%- set tool_call = tool_call.function %}
                    {%- endif %}
                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- if tool_call.arguments is defined %}
                        {%- for args_name, args_value in tool_call.arguments|items %}
                            {{- '<parameter=' + args_name + '>\n' }}
                            {%- set args_value = args_value if args_value is string else args_value | string %}
                            {{- args_value }}
                            {{- '\n</parameter>\n' }}
                        {%- endfor %}
                    {%- endif %}
                    {{- '</function>\n</tool_call>' }}
                {%- endfor %}
                {{- '<|im_end|>\n' }}
            {%- elif message.role == "user" or message.role == "system" or message.role == "assistant" %}
                {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
            {%- elif message.role == "tool" %}
                {%- if loop.previtem and loop.previtem.role != "tool" %}
                    {{- '<|im_start|>user\n' }}
                {%- endif %}
                {{- '<tool_response>\n' }}
                {{- message.content }}
                {{- '\n</tool_response>\n' }}
                {%- if not loop.last and loop.nextitem.role != "tool" %}
                    {{- '<|im_end|>\n' }}
                {%- elif loop.last %}
                    {{- '<|im_end|>\n' }}
                {%- endif %}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
            {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\n' }}
        {%- endif %}
        """#
}
