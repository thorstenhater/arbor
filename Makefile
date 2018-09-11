all: intercomm

intercomm: intercomm.cpp
	mpic++ -Wall intercomm.cpp -o intercomm

clean: ; rm intercomm
