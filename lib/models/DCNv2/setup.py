#!/usr/bin/env python

import os
import glob
import sys

# 在导入torch之前设置环境变量
# RTX 4060 需要 sm_89 架构，同时保留 sm_75 以兼容其他 GPU
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '7.5;8.9')

import torch

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": ["-std=c++17"]}  # PyTorch 2.0+ requires C++17
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        
        # 设置环境变量，包含 RTX 4060 的架构 (sm_89)
        # RTX 4060 需要 sm_89 架构支持
        # 同时保留 sm_75 以兼容其他 GPU
        os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5;8.9'
        
        # 获取CUDA版本信息
        print(f"CUDA_HOME: {CUDA_HOME}")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            # 添加更多编译选项以解决潜在问题
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-O3",
            # 明确指定架构：包含 sm_75 和 sm_89 (RTX 4060)
            "-gencode", "arch=compute_75,code=sm_75",
            "-gencode", "arch=compute_89,code=sm_89",
        ]
    else:
        raise NotImplementedError('Cuda is not available')

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "_ext",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

# 自定义BuildExtension类，完全禁用自动架构检测并改进错误报告
class BuildExtension(torch.utils.cpp_extension.BuildExtension):
    def __init__(self, *args, **kwargs):
        # 禁用ninja以获得更清晰的错误信息（如果ninja有问题）
        # 可以通过环境变量控制：USE_NINJA=0
        if 'USE_NINJA' in os.environ and os.environ['USE_NINJA'] == '0':
            kwargs['use_ninja'] = False
        super().__init__(*args, **kwargs)
    
    def build_extensions(self):
        print("=" * 60)
        print("开始编译DCNv2扩展模块...")
        print("=" * 60)
        
        # 在构建前，临时禁用自动架构检测和版本检查
        original_get_cuda_arch_flags = torch.utils.cpp_extension._get_cuda_arch_flags
        original_check_cuda_version = None
        
        # 尝试获取并替换版本检查函数
        if hasattr(torch.utils.cpp_extension, '_check_cuda_version'):
            original_check_cuda_version = torch.utils.cpp_extension._check_cuda_version
            def dummy_check_cuda_version(*args, **kwargs):
                # 跳过CUDA版本检查
                pass
            torch.utils.cpp_extension._check_cuda_version = dummy_check_cuda_version
        
        def dummy_get_cuda_arch_flags(*args, **kwargs):
            # 返回空列表，完全跳过自动检测
            # 架构已经在环境变量和nvcc参数中手动指定
            return []
        
        # 临时替换函数
        torch.utils.cpp_extension._get_cuda_arch_flags = dummy_get_cuda_arch_flags
        
        try:
            super().build_extensions()
            print("=" * 60)
            print("编译成功！")
            print("=" * 60)
        except (ValueError, RuntimeError) as e:
            error_msg = str(e)
            print("=" * 60)
            print("编译错误详情：")
            print(error_msg)
            print("=" * 60)
            
            # 处理 CUDA 版本不匹配错误
            if "mismatches" in error_msg.lower() or "CUDA version" in error_msg:
                print("\n检测到CUDA版本不匹配。")
                print("正在尝试绕过版本检查...")
                # 设置环境变量来绕过版本检查，包含 sm_89
                os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5;8.9'
                # 再次尝试，但这次直接调用编译命令而不经过版本检查
                try:
                    # 直接调用底层的编译方法
                    for ext in self.extensions:
                        self.build_extension(ext)
                    print("绕过版本检查后编译成功！")
                except Exception as e2:
                    print(f"绕过版本检查后编译仍然失败: {e2}")
                    # 如果还是失败，尝试升级PyTorch
                    print("\n建议：升级PyTorch到CUDA 12.1版本")
                    print("执行: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                    raise
            elif "Unknown CUDA arch" in error_msg or "8.9" in error_msg:
                print("\n检测到CUDA架构兼容性问题。")
                print("正在尝试使用兼容模式重新编译...")
                # 包含 sm_89 以支持 RTX 4060
                os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5;8.9'
                # 再次尝试
                try:
                    super().build_extensions()
                    print("兼容模式编译成功！")
                except Exception as e2:
                    print(f"兼容模式编译也失败: {e2}")
                    raise
            else:
                # 其他类型的错误，直接抛出
                print("\n完整的错误堆栈：")
                import traceback
                traceback.print_exc()
                raise
        except Exception as e:
            print("=" * 60)
            print(f"意外的编译错误: {type(e).__name__}: {e}")
            print("=" * 60)
            import traceback
            traceback.print_exc()
            raise
        finally:
            # 恢复原函数
            torch.utils.cpp_extension._get_cuda_arch_flags = original_get_cuda_arch_flags
            if original_check_cuda_version is not None:
                torch.utils.cpp_extension._check_cuda_version = original_check_cuda_version

setup(
    name="DCNv2",
    version="0.1",
    author="charlesshang",
    url="https://github.com/charlesshang/DCNv2",
    description="deformable convolutional networks",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
