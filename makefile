venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install --prefer-binary -Uqr requirements.txt
	touch venv/bin/activate

test: venv
	. venv/bin/activate && pytest

testprofile: venv
	. venv/bin/activate && python3 -m cProfile -o profile /usr/local/bin/pytest
	. venv/bin/activate && python3 -m gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

pylint.log: venv
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc syserol > pylint.log || echo "pylint exited with $?")

clean:
	rm -rf *.svg output venv pylint.log