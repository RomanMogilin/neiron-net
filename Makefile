

run:
	@python3 main.py
	@python3 main.py
	@python3 main.py
	@python3 main.py
	@python3 main.py
	@python3 main.py
	@python3 main.py
	@python3 main.py
	@python3 main.py
	@python3 main.py
	@if [ ! -d $(__pycache__) ] ; then rm -r __pycache__; fi
	@rm -r */__pycache__