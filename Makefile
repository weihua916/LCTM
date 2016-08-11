model: main.o LCTM.o dataset.o
	g++ -std=c++11 -O3 -o model main.o LCTM.o dataset.o

dataset.o: dataset.cpp Utils.h Matrix.h
	g++ -std=c++11 -O3 -c -o dataset.o dataset.cpp

LCTM.o: LCTM.cpp Utils.h Matrix.h
	g++ -std=c++11 -O3 -c -o LCTM.o LCTM.cpp

main.o: main.cpp Utils.h Matrix.h
	g++ -std=c++11 -O3 -c -o main.o main.cpp
	
clean:
	rm -f model main.o LCTM.o dataset.o
