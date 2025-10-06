import json


# Чтение json-файла
def read_params(params_file_path: str) -> dict:
    with open(params_file_path) as fin:
        content = fin.read()
    data_dict = json.loads(content)
    return data_dict
    
    