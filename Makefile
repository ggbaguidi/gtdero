MAKE_FILE=makefile.solar
all: main

.PHONY: clean

main:
	echo "============================ GT_DERO ========================"

demand:
	make -f makefile.demand

solar:
	make -f makefile.solar

run_week:
	for i in $(shell seq 1 7); do \
		make -f $(MAKE_FILE); \
	done

clean:
	rm -rf checkpoints/* *results result*.txt
