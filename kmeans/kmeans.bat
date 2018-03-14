del *.png
del *.ppm
cd ..\x64\Release
del *.png
del *.ppm
kmeans.exe 0 "..\..\QtCuda\~screenshot_0.ppm" 16
xcopy /f /y *.png ..\..\kmeans
xcopy /f /y *.ppm ..\..\kmeans
pause