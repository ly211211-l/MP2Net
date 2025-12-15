**环境配置：**



**创建激活虚拟环境——**

conda create -n mp2net python=3.6 -y

conda activate mp2net



**安装pytorch1.7\&cuda10.2（不兼容）**

conda install pytorch==1.7.0 torchvision cudatoolkit=10.2 -c pytorch -y

python -c "import torch; print(torch.\_\_version\_\_); print(torch.cuda.is\_available())"   测试输出（1.7.0 True）



**删除环境**：conda remove -n mp2net --all -y

**创建新环境（Python 3.9）：**

conda create -n mp2net python=3.9 -y

conda activate mp2net



**安装支持 RTX4060 的 PyTorch（PyTorch 2.4.0 + cu121）：**

conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=12.1 -c pytorch -c nvidia

\*\*测试 CUDA 是否可用：\*\*python -c "import torch; print(torch.\_\_version\_\_); print(torch.cuda.is\_available())"



**安装常用依赖：**

pip install cython opencv-python pillow scipy matplotlib tqdm shapely

conda install -c conda-forge pycocotools

