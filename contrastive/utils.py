
from typing import List,Optional,Tuple,Dict

import pandas as pd
from transformers import PreTrainedTokenizer

def __dataframe_info_simple(varname:str, table: pd.DataFrame, placeholder_value:str) -> str:
    
    desc_info_lines = [
        (f"- {placeholder_value} \'{col}\' {table[col].dtype}")
        for col in table.columns
    ]

    desc_info = "\n".join(desc_info_lines)
    
    return f"/*\nDetails about the '{varname}' dataframe that can be used as follows:\n{desc_info}\n*/"


def __build_table_question(tables: Dict[str,pd.DataFrame],table_placeholder_value):
    
    df_info_list = [__dataframe_info_simple(varname,table,table_placeholder_value) for varname,table in tables.items()]
    return '\n'.join(df_info_list)


def _build_table_info(tables:Dict[str,pd.DataFrame],table_placeholder_value:str):
    # should warning
    if not tables or table_placeholder_value is None:
        return ""

    return __build_table_question(tables,table_placeholder_value)



def convert_history_to_messages(history:List[Tuple[str,str]]):
    return [
        {"role":k,"content":content} for k, content in history
    ]


def build_prompt(query:str,
                 tables:Dict[str,pd.DataFrame], 
                 tokenizer:PreTrainedTokenizer,
                 table_place_holder_val:Optional[str] =None,
                 history:Optional[List] = None) -> Tuple[List,str]:

    if history is None:
        history = [(
        "system","""You are TableGPT, an expert Python data analyst developed by 浙江大学计算机创新技术研究院 (Institute of Computer Innovation of Zhejiang University, or ZJUICI). Your job is to help user analyze datasets by writing Python code. Each markdown codeblock you write will be executed in an IPython environment, and you will receive the execution output. You should provide results analysis based on the execution output.
For politically sensitive questions, security and privacy issues, or other non-data analyze questions, you will refuse to answer."""
    )]
    
    orignal_prompt = tokenizer.apply_chat_template(
            convert_history_to_messages(history),
            add_generation_prompt=True,
            tokenize=False
        )

    if table_place_holder_val in orignal_prompt:
        history.append(
            (
                "user", query
            )
        )
        return history,tokenizer.apply_chat_template(
            convert_history_to_messages(history),
            add_generation_prompt=True,
            tokenize=False
        )
    
    if not tables:
        history.append(
            (
                "user", query
            )
        )
        return history, tokenizer.apply_chat_template(
            convert_history_to_messages(history),
            add_generation_prompt=True,
            tokenize=False
        )

    else:
        history.append(
            (
                "system", _build_table_info(tables,table_place_holder_val)
            )
        )

        history.append(
            (
                "user", query
            )
        )

    return history, tokenizer.apply_chat_template(
        convert_history_to_messages(history),
        add_generation_prompt=True,
        tokenize=False
    )