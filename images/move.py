import shutil, os

def move_file(begin_fold, train_fold, val_fold, pattern):

    #obtain the files necessary
    print("C")
    files = []
    dir_list = os.listdir(begin_fold)
    for path in dir_list:
        rel_path = path.strip("./")
        if os.path.isfile(os.path.join(begin_fold, path)) and pattern in rel_path:
            files.append(rf"{path}")
    
    #move the files
    print("a")
    print(type(files))
    files_len = len(files)
    print(type(files_len))
    train_num = (7/10) * files_len 
    train_set = files[:train_num]
    val_set = files[train_num:]
    for t_path in train_set:
        start_abs = os.path.join(begin_fold, t_path)
        end_abs = os.path.join(train_fold, t_path)
        shutil.move(start_abis, end_abis)
    for v_path in val_set:
        start_abs = os.path.join(begin_fold, v_path)
        end_abs = os.path.join(val_fold, v_path)
        shutil.move(start_abis, end_abis)
move_file(
    r"C:\Users\knott\robotics_limecomputer\images\not_is_cone", 
    r"C:\Users\knott\robotics_limecomputer\images\validation\not_is_cone",
    r"C:\Users\knott\robotics_limecomputer\images\train\not_is_cone", 
    "")