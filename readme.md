# Mac Install
### Setup JDK for MineDojo
See [docs](https://docs.minedojo.org/sections/getting_started/install.html#prerequisites) for details

#### Step 1: Python
Setup conda env
```shell
conda create -n MinecraftRL python=3.9
```
Activate conda env
```shell
conda activate MinecraftRL
```
Install dependencies
```shell
pip install -r requirements.txt
```

#### Step 2: Java
Install Temurin JDK 8
```shell
brew tap homebrew/cask-versions
brew install --cask temurin8
```
Check where the temurin JDK went
```shell
/usr/libexec/java_home -V
```
This should print something like a table of every JDK you have and it's path. Keep note of the path of Temurin 8.
For this example let's say its `/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home`
```shell
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home
```

#### Step 3: Running
With a terminal at the project location, run
```shell
python main.py
```
