.PHONY : tests

all :
	echo "Configure your own targets here."

tests :
	py.test -v --cov variational_bayes --cov-report html
