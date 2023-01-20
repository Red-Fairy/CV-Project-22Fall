import numpy as np
import os
import os.path as osp
import shutil

def main():
    for filename in os.listdir(r'labels\train'):
        z=0
        f=open(r'labels\\train\\'+filename,'r')
        str1=f.read
        if not os.path.getsize(r'labels\\train\\'+filename):
            z=1
        f.close()
        if z==0 :
            continue
        os.remove(r'labels\\train\\'+filename)
        filename2 = list(filename)
        for i in range(0,len(filename2)):
            if filename2[i]=='.' :
                filename2[i+1]='j'
                filename2[i + 2] = 'p'
                filename2[i + 3] = 'g'
                break
        for filename1 in os.listdir(r'images\train'):
            filename3=list(filename1)
            if filename3 == filename2 :
                os.remove(r'images\\train\\'+filename1)
                continue



    for filename in os.listdir(r'labels\val'):
        z=0
        f=open(r'labels\\val\\'+filename,'r')
        str1=f.read
        if not os.path.getsize(r'labels\\val\\' + filename):
            z=1
        f.close()
        if z==0 :
            continue
        os.remove(r'labels\\val\\'+filename)
        filename2 = list(filename)
        for i in range(0,len(filename2)):
            if filename2[i]=='.' :
                filename2[i+1]='j'
                filename2[i + 2] = 'p'
                filename2[i + 3] = 'g'
                break
        for filename1 in os.listdir(r'images\val'):
            filename3=list(filename1)
            if filename3 == filename2 :
                os.remove(r'images\\val\\'+filename1)
                continue


if __name__ == '__main__':
    main()
