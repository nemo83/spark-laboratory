MAKEFLAGS = -s

.PHONY: movielens

movielens:
	curl -s -O "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
	unzip ml-latest-small.zip -d src/main/resources
	rm ml-latest-small.zip
