.PHONY : tests clean data/sp100/yahoo data/sp100/google

NOTEBOOKS = $(wildcard examples/*.ipynb)
NOTEBOOK_OUTPUTS = $(NOTEBOOKS:.ipynb=.html)

examples : $(NOTEBOOK_OUTPUTS)

$(NOTEBOOK_OUTPUTS) : %.html : %.ipynb
	jupyter nbconvert --execute --ExecutePreprocessor.timeout=None --allow-errors $<

clean :
	rm examples/*.html

tests :
	py.test -v --cov elboflow --cov-report html -rsx

data/sp100/yahoo: data/sp100/yahoo/sp100_symbols_2016-12-30.txt
	scripts/download.py -o data/sp100/yahoo -l $(<:.txt=.log) -f $< yahoo 2007-01-01 2017-01-01

data/sp100/google: data/sp100/google/sp100_symbols_2016-12-30.txt
	scripts/download.py -o data/sp100/google -l $(<:.txt=.log) -f $< google 2007-01-01 2017-01-01
