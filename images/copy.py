import os, glob
print("a")

def remove(pathq, pattern):
    for f in os.listdir("not_is_cone"):
        if pattern in f:
            os.remove("not_is_cone/"+f)

remove("./not_is_cone", "copy")
        
print("B")