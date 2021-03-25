
flist = 2 3 4 5 6 S1 S2

all: $(patsubst %, output/figure%.svg, $(flist))

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install --prefer-binary -Uqr requirements.txt
	touch venv/bin/activate
	mkdir -p output
	touch -m output/figure*.svg

output/figure%.svg: venv genFigure.py
	mkdir -p output
	. venv/bin/activate && ./genFigure.py $*

test: venv
	. venv/bin/activate && pytest -s -v -x

output/manuscript.md: venv manuscript/*.md
	. venv/bin/activate && manubot process --content-directory=manuscript --output-directory=output --cache-directory=cache --skip-citations --log-level=INFO
	cp -r manuscript/images output/
	git remote rm rootstock

output/manuscript.html: venv output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/html.yaml \
		output/manuscript.md

output/manuscript.docx: venv output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/docx.yaml \
		output/manuscript.md

clean:
	rm -rf output venv pylint.log
