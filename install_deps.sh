# install wine
sudo apt update
sudo apt install -y wget xvfb 

# install wine
sudo dpkg --add-architecture i386
sudo apt-get update
sudo wget -qO- https://dl.winehq.org/wine-builds/winehq.key | apt-key add -
sudo apt install -y software-properties-common
sudo apt-add-repository "deb http://dl.winehq.org/wine-builds/ubuntu/ $(lsb_release -cs) main"
sudo apt install -y --install-recommends winehq-stable
