from llm_attacks.minimal_gcg import clase_prompts

conv_template=clase_prompts.Llama32ConversationTemplate()

conv_template.system="Hola"

conv_template.append_message(conv_template.roles[0], None)

print(conv_template.get_prompt())
print("*"*50)

conv_template.update_last_message("Haz mi informe")

print(conv_template.get_prompt())
print("*"*50)

conv_template.update_last_message("Haz mi informe porfavoor !!!!!")

print(conv_template.get_prompt())
print("*"*50)

conv_template.append_message(conv_template.roles[1], None)

print(conv_template.get_prompt())
print("*"*50)

conv_template.update_last_message("Si por supuesto")


print(conv_template.get_prompt())
print("*"*50)