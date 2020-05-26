---
title:  配置VS Code - 进阶
categories:
- computer-skills
---

&emsp;&emsp;前段时间服役了多年的macbook突然宕机了，发现是硬盘坏了，彻底坏了的那种，遂重新装机，重新配环境。在配环境的过程中发现很多低端重复性的操作简直是浪费时间，尤其是基于vscode这种高自由度IDE的环境配置，所以在此做一些记录，希望以后在迁徙生产力的时候能少费功夫。主要有vscode上C++、OpenCV、code runner、setting sync、右键从文件夹进入等配置。
<!-- more -->

***
>+ # C++任务配置

&emsp;&emsp;macOS系统上如果提前装好了Xcode，那么默认的clang编译器也装好了，可以用以下命令行来检验：
```shell
clang --version
``` 
&emsp;&emsp;然后需要在vscode中安装微软官方的`C/C++`插件。接下来是配置默认生成任务。在C++源代码文件夹中打开代码文件，保持窗口激活，在菜单`"终端-->配置默认生成任务"`中点击，会出现如下的下拉菜单，选择"clang++ build active file"。
![](/assets/images/vscode-cpp/1.png)
&emsp;&emsp;选择后，在文件夹的根路径下会自动生成`.vscode`文件夹，其中自动生成了`tasks.json`配置文件，具体如下：
```json
{
  // See https://go.microsoft.com/fwlink/?LinkId=733558
  // for the documentation about the tasks.json format
  "version": "2.0.0",
  "tasks": [
    {
      "type": "shell",
      "label": "clang++ build active file",
      "command": "/usr/bin/clang++",
      "args": [
        "-std=c++17",
        "-stdlib=libc++",
        "-g",
        "${file}",
        "-o",
        "${fileDirname}/${fileBasenameNoExtension}"
      ],
      "options": {
        "cwd": "${workspaceFolder}"
      },
      "problemMatcher": ["$gcc"],
      "group": {
        "kind": "build",
        "isDefault": true
      }
    }
  ]
}
```
&emsp;&emsp;通过此文件，已经指定了编译器及相关参数，在菜单`"终端-->运行生成文件"`中点击即可对源码进行编译，编译后在同路径下会生成同名可执行文件，`./file_name`即可执行。（如果有多个源文件需要编译，则更改变量`${file}`为`${workspaceFolder}/*.cpp`即可。）  
&emsp;&emsp;上述配置仅仅可用于编译执行，当需要debug时，还需要生成`.vscode`下`launch.json`配置文件。保持代码窗口激活，在菜单`"运行-->添加配置"`中点击，出现下拉菜单，依次选择`"C++ (GDB/LLDB)-->clang++ build and debug active file"`，会在`.vscode`文件夹中自动生成`launch.json`配置文件：
```json
{
  // Use IntelliSense to learn about possible attributes.
  // Hover to view descriptions of existing attributes.
  // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
  "version": "0.2.0",
  "configurations": [
    {
      "name": "clang++ - Build and debug active file",
      "type": "cppdbg",
      "request": "launch",
      "program": "${fileDirname}/${fileBasenameNoExtension}",
      "args": [],
      "stopAtEntry": true,
      "cwd": "${workspaceFolder}",
      "environment": [],
      "externalConsole": false,
      "MIMode": "lldb",
      "preLaunchTask": "clang++ build active file"
    }
  ]
}
```
&emsp;&emsp;由此便可以debug了，在菜单`"运行-->启动调试"`中点击，即能激活左边栏的debug界面，可以设置断点和逐步运行等调试步骤。  
&emsp;&emsp;前面的可编译和可调试还不够，当需要链接头文件时，需要指定头文件的路径，因此还需要生成`.vscode`下`c_cpp_properties.json`配置文件。依然是保持源代码窗口激活，在菜单`"查看-->命令面板"`中点击，手动输入`"C/C++: Edit Configurations (UI)"`并选中，会自动生成`.vscode`下`c_cpp_properties.json`文件，内容如下：
```json
{
  "configurations": [
    {
      "name": "Mac",
      "includePath": ["${workspaceFolder}/**"],
      "defines": [],
      "macFrameworkPath": [
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks"
      ],
      "compilerPath": "/usr/bin/clang",
      "cStandard": "c11",
      "cppStandard": "c++17",
      "intelliSenseMode": "clang-x64"
    }
  ],
  "version": 4
}
```
&emsp;&emsp;其中，`"includePath"`即是头文件的路径，可手动添加外部库文件。  
&emsp;&emsp;由此，C++在VSCode中的配置基本完毕，当需要新开项目文件夹时，直接将`".vscode"`文件夹整体复制迁移至该文件夹的根路径下即可。

<br
/>
&emsp;&emsp;官网的配置指南：[Configure VS Code for Clang/LLVM on macOS](https://code.visualstudio.com/docs/cpp/config-clang-mac)

***
>+ # OpenCV安装与配置

&emsp;&emsp;如果只在python环境中使用opencv的话，直接`pip install opencv-python`就好了。但是如果需要在C++中使用，需要官网下载源代码，用cmake编译安装。  
&emsp;&emsp;首先，确保系统已安装cmake，没有的话，可以使用Homebrew安装：
```shell
brew install cmake
```
&emsp;&emsp;然后，在opencv官网下载源包，另创建一个空的临时编译文件夹`build_opencv`，在该文件夹路径下，配置源文件：
```shell
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON ./path-to-opencv
```
&emsp;&emsp;随后，编译并安装：
```shell
make -j7 # 开七个线程
sudo make install
```
&emsp;&emsp;安装完之后，库文件在`/usr/local/lib`下，头文件在`/usr/local/include`下。如果C++代码中包含头文件`<opencv2/opencv.hpp>`之类的，需要在`.vscode/c_cpp_properties.json`的`"includePath"`中添加路径：`/usr/local/include`。同时，为了能正确找到库文件，还要在`.vscode/tasks.json`的`"args"`中添加：
```json
    "-I", "/usr/local/include",
    "-I", "/usr/local/include/opencv",
    "-I", "/usr/local/include/opencv2",
    "-L", "/usr/local/lib",
    "-l", "opencv_core",
    "-l", "opencv_imgproc",
    "-l", "opencv_imgcodecs",
    "-l", "opencv_video",
    "-l", "opencv_ml",
    "-l", "opencv_highgui",
    "-l", "opencv_objdetect",
    "-l", "opencv_flann",
    "-l", "opencv_imgcodecs",
    "-l", "opencv_photo",
    "-l", "opencv_videoio"  
```
&emsp;&emsp;至此，opencv库应该就可以在vscode中正常使用了。

***
>+ # Code Runner一键运行

&emsp;&emsp;前面讲了每次运行代码都需要先编译再打开终端运行，过程比较繁琐。插件`Code Runner`很好地解决了这个麻烦，配置好一键运行就成，适合于菜鸡刷题使用，让小白在入门学习阶段把主要精力放在代码内容上，而不被繁琐的工程化问题所耽搁。  
&emsp;&emsp;针对具体的编译要求，需要对默认设置做相应修改。例如含有opencv库的C++代码，需要在系统设置文件`setting.json`中，加入以下设置：
```json
    "code-runner.runInTerminal": true,
    "code-runner.saveAllFilesBeforeRun": true,
    "code-runner.executorMap": {
        "cpp": "cd $dir && clang++ $fileName -o $fileNameWithoutExt -I /usr/local/include -I /usr/local/include/opencv -I /usr/local/include/opencv2 -L /usr/local/lib -l opencv_core -l opencv_imgproc -l opencv_imgcodecs -l opencv_video -l opencv_ml -l opencv_highgui -l opencv_objdetect -l opencv_flann -l opencv_imgcodecs -l opencv_photo -l opencv_videoio && ./$fileNameWithoutExt",
    },
```
&emsp;&emsp;像这样配置完成后，直接右键`Run Code`或者点击右上角的运行小按键（推荐），程序就自动编译并运行。
![](/assets/images/vscode-cpp/2.jpg)

***
>+ # VS Code 配置云端同步

&emsp;&emsp;vscode虽好，但每次碰上生产力迁移的时候，都需要在新电脑上重新配置一遍插件和设置，属实麻烦。那有没有像Chrome那样的账号同步功能，能把原先所配好的插件和设置都在云端保存下来呢？答案是有的，插件`Settings Sync`。
![](/assets/images/vscode-cpp/3.png)
&emsp;&emsp;它直接同Github账号相绑定，第一次上传配置时需要认证Github账户，没有Gist时首次上传会自动生成一个新的Gist ID。所有的操作（上传/下载/选项等）都可以在操作面板里搜索`sync`得到，反正我是记不住复杂的组合快捷键。当需要在新电脑上配置vscode时，直接输入Gist ID就可以下载原先的配置。

***
>+ # 右键从当前路径打开vscode

&emsp;&emsp;在windows中，可以在选中源代码文件夹后，右键选择在vscode中打开，打开后即为当前路径。但是macOS中，右键菜单权限比较高，无法直接右键进入。好在可以自定义”快速操作“，也能达到类似的效果。  
&emsp;&emsp;首先，打开 `Automator.app（自动操作）`，点击文件新建，选择`Quick Action（快速操作）`。
![](/assets/images/vscode-cpp/4.jpeg)
&emsp;&emsp;然后配置工作流，选择`文件或文件夹`
位于`访达`。左侧条目中选择`打开访达项目`，打开方式选择`Visual Studio Code`，保存为`Open in VSCode`。
![](/assets/images/vscode-cpp/5.jpeg)
&emsp;&emsp;随后就可以像在windows系统中右键选择进入一样，直接在当前路径下打开项目了。类似地，还可以配置在当前路径下打开终端，也非常实用。
![](/assets/images/vscode-cpp/6.png)

<br
/>
&emsp;&emsp;参考资料：https://liam.page/2020/04/22/Open-in-VSCode-on-macOS/
