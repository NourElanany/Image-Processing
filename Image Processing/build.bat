@echo off
echo Building Image Processor...

:: Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

:: Run PyInstaller
pyinstaller --name=improcess ^
            --onedir ^
            --windowed ^
            --icon=improcess.ico ^
            --hidden-import=PIL._tkinter_finder ^
            --hidden-import=customtkinter ^
            --hidden-import=tkinterdnd2 ^
            --hidden-import=scipy.ndimage ^
            --hidden-import=scipy.fft ^
            --collect-data=customtkinter ^
            --collect-data=tkinterdnd2 ^
            --add-data=".venv\Lib\site-packages\tkinterdnd2\tkdnd;tkinterdnd2/tkdnd" ^
            --add-data="alg;alg" ^
            --noupx ^
            --clean ^
            main.py

:: Check if build was successful
if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

echo Build completed successfully!
echo Executable is located in the dist/improcess folder

:: Clean up build files (optional)
echo Cleaning up build files...
rmdir /s /q build
del /q improcess.spec

pause
