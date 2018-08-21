HYPERPARAMETERS = $(shell ls hyperparameters*.json)
OUTPUT = $(HYPERPARAMETERS:.json=.html)

all : $(OUTPUT)

$(OUTPUT) : %.html : %.json latent-factor-clustering.ipynb
	HYPERPARAMETERS=$< jupyter nbconvert --execute --allow-errors --ExecutePreprocessor.timeout=None latent-factor-clustering.ipynb --output $@