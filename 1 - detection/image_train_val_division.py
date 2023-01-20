import numpy as np
import os
import os.path as osp
import shutil

def main():
    for filename in os.listdir(r'labels\train'):
        filename2 = list(filename)
        for i in range(0,len(filename2)):
            if filename2[i]=='.' :
                filename2[i+1]='j'
                filename2[i + 2] = 'p'
                filename2[i + 3] = 'g'
                break
        for filename1 in os.listdir(r'images'):
            filename3=list(filename1)
            if filename3 == filename2 :
                oldfile = "images\\"
                oldfile = os.path.join(oldfile, filename1)
                # oldfile="images\\%s" % (filename3)
                newfile='images\\train\\'
                newfile = os.path.join(newfile, filename1)
                # newfile='image\\train\\%s'%(filename3)
                shutil.copyfile(oldfile,newfile)



    for filename in os.listdir(r'labels\val'):
        filename2 = list(filename)
        for i in range(0,len(filename2)):
            if filename2[i]=='.' :
                filename2[i+1]='j'
                filename2[i + 2] = 'p'
                filename2[i + 3] = 'g'
                break
        for filename1 in os.listdir(r'images'):
            filename3=list(filename1)
            if filename3 == filename2 :
                oldfile = "images\\"
                oldfile = os.path.join(oldfile, filename1)
                # oldfile="images\\%s" % (filename3)
                newfile='images\\val\\'
                newfile = os.path.join(newfile, filename1)
                # newfile='image\\train\\%s'%(filename3)
                shutil.copyfile(oldfile,newfile)


if __name__ == '__main__':
    main()
