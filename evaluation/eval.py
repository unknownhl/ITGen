import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys

sys.path.append('../')
sys.path.append('../python_parser')

retval = os.getcwd()

import javalang
import torch
import json
from numpy import *
from python_parser.run_parser import extract_dataflow

def get_code_tokens(code):
    tokens = javalang.tokenizer.tokenize(code)
    code_tokens = [token.value for token in tokens]
    return code_tokens

device = torch.device("cuda")

model_list = ["CodeBERT_adv"]
task_list = ["Clone-detection"]
attack_list = ["alert", "beam", "itgen"]
# 0.4478
for model in model_list:
    for task in task_list:
        for attack_method in attack_list:
            adv_file = "../{}/{}/attack/result/attack_{}_all.jsonl".format(model, task, attack_method)
            attack = adv_file.split(".")[-2].split("_")[-1]
            code_list = []
            adv_code_list = []
            replaced_list = []
            query_list = []
            time_list = []
            total_identifiers_list = []
            total_token_list = []
            type_list = []
            with open(adv_file) as f:
                for line in f:
                    js = json.loads(line.strip())
                    code_list.append(js["Original Code"])
                    adv_code_list.append(js["Adversarial Code"])
                    replaced_list.append(js["Replaced Identifiers"])
                    query_list.append(js["Query Times"])
                    time_list.append(js["Time Cost"])
                    total_identifiers_list.append(js["Identifier Num"])
                    total_token_list.append(js["Program Length"])
                    type_list.append(js["Type"])
            skip_var_num = 0
            succ_num = 0

            replece_var_list = []
            replece_token_list = []

            succ_query_list = []
            succ_time_list = []
            for index, type in enumerate(type_list):

                if type != '0':
                    succ_query_list.append(query_list[index])
                    succ_time_list.append(time_list[index])
                    orginal_code = code_list[index]
                    adv_code = adv_code_list[index]
                    _, _, code1_tokens = extract_dataflow(orginal_code, "java")
                    try:
                        _, _, code2_tokens = extract_dataflow(adv_code, "java")
                        code2_tokens = ['"\\n"' if i == '"\n"' else i for i in code2_tokens]
                    except Exception as e:
                        skip_var_num += 1
                        continue

                    succ_num += 1
                    replaced_info = replaced_list[index]
                    replaced_info = replaced_info.split(",")[:-1]
                    replace_iden = []
                    for replace in replaced_info:
                        replace_pair = replace.split(":")
                        if replace_pair[0] != replace_pair[1]:
                            replace_iden.append(replace_pair[0])
                    replece_var_list.append(len(replace_iden))
                        #
                    sum_token = 0
                    for tok in code1_tokens:
                        if tok in replace_iden:
                            sum_token += 1
                    replece_token_list.append(sum_token)
                    
            print("*"*30)
            print("{}-{}-{}".format(model, task, attack_method))
            print("ASR: {:.2%}".format(succ_num/len(code_list)))
            print("AMQ: {:.2f}".format(mean(succ_query_list)))
            print("ART: {:.2f}".format(mean(succ_time_list)))