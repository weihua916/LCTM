model: main.o LDA.o LDA_CVB0.o dataset.o
	g++ -std=c++11 -O3 -o model main.o LDA.o LDA_CVB0.o dataset.o

dataset.o: dataset.cpp Utils.h Matrix.h
	g++ -std=c++11 -O3 -c -o dataset.o dataset.cpp

LDA.o: LDA.cpp Utils.h Matrix.h
	g++ -std=c++11 -O3 -c -o LDA.o LDA.cpp

LDA_CVB0.o: LDA_CVB0.cpp Utils.h Matrix.h
	g++ -std=c++11 -O3 -c -o LDA_CVB0.o LDA_CVB0.cpp

main.o: main.cpp Utils.h Matrix.h
	g++ -std=c++11 -O3 -c -o main.o main.cpp
	
clean:
	rm -f model main.o LDA.o LDA_CVB0.o dataset.o
