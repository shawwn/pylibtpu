

libtpu_client: libtpu_client.c
	gcc -o libtpu_client libtpu_client.c -ldl

test: libtpu_client
	@sudo ./libtpu_client
