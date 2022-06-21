import os
import time
import json
import pickle
import shutil


def json_load(json_filepath):
    with open(json_filepath, 'r') as f:
        data = json.load(f)
        return data

def json_dump(values, file_path=None):
    if file_path is None:
        print(json.dumps(values, sort_keys=True, indent=4, separators=(',', ': ')))
    else:
        with open(file_path, 'w') as outfile:
            json.dump(values, outfile,  sort_keys=True, indent=4, separators=(',', ': '))

def get_name_with_time(config):

    timestamp = time.strftime("%Y%m%d-%H%M%S")


    if config["debug_mode"]:
        save_folder_name = "debug_mode"
    else:
        save_folder_name = ""


    save_folder_name = os.path.join(save_folder_name,timestamp)

    return save_folder_name

def create_working_folder(config=None):
    print(os.getcwd())
    if not config:
        # Load Configuration File
        json_config_file_path = r'config/config.json'
        config = json_load(json_config_file_path)


    # create working dir string
    workdir = get_name_with_time(config)
    current_work_dir = os.path.join(config["output_directory"], workdir)

    # create folders
    os.makedirs(current_work_dir, exist_ok=True)
    os.chmod(current_work_dir, mode=0o777)
    os.makedirs(os.path.join(current_work_dir, 'Models'), exist_ok=True)
    os.chmod(os.path.join(current_work_dir, 'Models'), mode=0o777)
    print(current_work_dir)

    # update opt with working dir
    config["workdir"] = current_work_dir

    # save configuration file in working dir
    json_dump(config, file_path=os.path.join(current_work_dir, "config.json"))


    return current_work_dir


if __name__ == '__main__':

    pass