import subprocess
import time
from os.path import join
import os.path
from sys import argv
import os


def main(objext: str, output_dir: str, args: str, block: bool, cudaid=0):
    """
    objext:目标文件(有没有扩展名都可以)
    output_dir:输出目录
    args:目标文件需要的参数
    block:是否阻塞等待子进程结束
    cudaid:指定cudaid
    """
    # obj目标文件,ext扩展名
    obj, ext = os.path.splitext(objext)
    if ext == "":
        objext = obj+".py"

    output = time.strftime(f"{obj}-%m-%d[%H:%M:%S]", time.localtime()) + ".log"
    output = join(output_dir, output)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 剩余参数传给目标文件
    cmd = f"CUDA_VISIBLE_DEVICES={cudaid} python -u {objext} {args} >{output} 2>&1"
    # 非阻塞在后台执行
    if not block:
        cmd = f"CUDA_VISIBLE_DEVICES={cudaid} nohup python -u {objext} {args} >{output} 2>&1 &"
    print(cmd)
    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    # 直接启动脚本不会阻塞
    args = " ".join(argv[3:])

    main(argv[1], argv[2], args, block=False)
