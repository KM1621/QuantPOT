@echo off

rem set CONDAPATH=/cygdrive/d/SOFTWARES/Anaconda_Installed/envs/pytorch_nemo_07/python
set CONDAPATH=D:\SOFTWARES\Anaconda_Installed

rem Define here the name of the environment
set ENVNAME=pytorch_nemo_07

rem The following command activates the base environment.
rem call C:\ProgramData\Miniconda3\Scripts\activate.bat C:\ProgramData\Miniconda3
if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)
rem Activate the conda environment
rem Using call is required here, see: https://stackoverflow.com/questions/24678144/conda-environments-and-bat-files
call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

rem Run a python script in that environment
rem python NEMO_main_CIFAR10.py --ptq True --epoch 25 --lr 0.0001 --init mnist_cnn_fp.pt --qat True --bit %%q  --pretrain True
FOR %%q IN (2 3 4 5 6 7 8 9 10 11 12 13) DO (
		python NEMO_main_CIFAR10.py --epoch 150 --qat True --ptq True --lr 0.001 --init ./CIFAR_cnn_fp.pth --bit %%q
    )
rem Deactivate the environment
call conda deactivate

pause