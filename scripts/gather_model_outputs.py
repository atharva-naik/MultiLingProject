import os
import shutil

def list_files(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith(".json"): continue
            if not file.startswith("test_"): continue
            file_list.append(os.path.join(root, file))

    return file_list

# main
if __name__ == "__main__":
    files = list_files("hf_out")
    out_folder = "model_outputs"
    os.makedirs(out_folder, exist_ok=True)
    for file in files:
        src = file
        dest = os.path.join(out_folder, file.replace("/","-"))
        print(src, dest)
        shutil.copyfile(src, dest)