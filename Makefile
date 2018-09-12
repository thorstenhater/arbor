all: intercomm

intercomm: intercomm.cpp
	mpic++ -Wall -g intercomm.cpp -o intercomm

clean: ; rm intercomm
