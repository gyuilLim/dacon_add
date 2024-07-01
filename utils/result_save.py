import json


def result_save(path, loss_info) :
    with open(path + '/loss_info.json', 'w') as json_file :
        json.dump(loss_info, json_file)