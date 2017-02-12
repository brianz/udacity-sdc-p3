.PHONY:	all drive

all:
	docker run --rm -it \
		-p 8888:8888 -v `pwd`:/src \
		udacity/carnd-term1-starter-kit

#python drive.py model.h5
drive :
	docker run -it --rm \
		-p 4567:4567 \
		-v `pwd`:/src \
		udacity/carnd-term1-starter-kit \
		bash
