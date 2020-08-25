
flist = 1 2 3 4 5 6 7 8 9 10

all: pylint.log $(patsubst %, figure%.svg, $(flist))

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install --prefer-binary -Uqr requirements.txt
	touch venv/bin/activate

figure%.svg: venv genFigure.py syserol/figures/figure%.py
	. venv/bin/activate && ./genFigure.py $*

test: venv
	. venv/bin/activate && pytest

testprofile: venv
	. venv/bin/activate && python3 -m cProfile -o profile /usr/local/bin/pytest
	. venv/bin/activate && python3 -m gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

pylint.log: venv
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc syserol > pylint.log || echo "pylint exited with $?")

output/manuscript.md: venv manuscript/*.md
	. venv/bin/activate && manubot process --content-directory=manuscript --output-directory=output --cache-directory=cache --skip-citations --log-level=INFO
	git remote rm rootstock

output/manuscript.html: venv output/manuscript.md $(patsubst %, figure%.svg, $(flist))
	mkdir output/output
	cp *.svg output/
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/html.yaml

clean:
	rm -rf *.svg output venv pylint.log
