all: intercomm arbor_proxy

intercomm: intercomm.cpp
	mpic++ -Wall -g intercomm.cpp -o intercomm

arbor_proxy: arbor_proxy.cpp
	mpic++ -Wall -g arbor_proxy.cpp -o arbor_proxy

clean: ; rm intercomm arbor_proxy
