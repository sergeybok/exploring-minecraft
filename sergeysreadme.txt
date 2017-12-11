
use python2.7

export JAVA_HOME=`/usr/libexec/java_home -v 1.8`

mkdir build; cd build;

cmake ..

comment out everything with CSharp


export MALMO_XSD_PATH=~/MalmoPlatform/Schemas

make
https://github.com/Microsoft/malmo/blob/master/doc/build_macosx.md (if mission.h file can't be found)

wget https://raw.githubusercontent.com/bitfehler/xs3p/1b71310dd1e8b9e4087cf6120856c5f701bd336b/xs3p.xsl -P ./Schemas


run tabular_q_learning.py from examples, it seems to work


need to ./launchClient.sh in one terminal, and run python script in other



build/install/Python_Examples <== directory where I run shit






