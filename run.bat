@echo off
REM 设置环境变量解决OpenMP冲突
set KMP_DUPLICATE_LIB_OK=TRUE

REM 显示当前配置
echo ===== 图像描述生成模型运行脚本 =====
echo 环境变量已设置: KMP_DUPLICATE_LIB_OK=TRUE
echo.

REM 根据参数运行不同的脚本
if "%1"=="train" (
    echo 开始训练模型...
    python main.py --mode train --num_epochs 5 --batch_size 32
) else if "%1"=="test" (
    echo 开始测试模型...
    python main.py --mode test
) else if "%1"=="eval" (
    echo 开始评估模型...
    python eval.py --split test
) else if "%1"=="demo" (
    echo 展示数据集样本...
    python dataset.py
) else (
    echo.
    echo 用法:
    echo   run.bat train    - 训练模型
    echo   run.bat test     - 测试模型
    echo   run.bat eval     - 评估模型
    echo   run.bat demo     - 展示数据集
    echo.
    echo 或者直接运行: python 脚本名.py
) 