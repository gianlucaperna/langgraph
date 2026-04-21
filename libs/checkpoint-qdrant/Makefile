.PHONY: test test_watch lint type format

######################
# TESTING AND COVERAGE
######################

start-qdrant:
	docker compose -f tests/compose-qdrant.yml up -V --force-recreate -d || ( \
		echo "Failed to start Qdrant, printing logs..."; \
		docker compose -f tests/compose-qdrant.yml logs; \
		exit 1 \
	)
	@echo "Waiting for Qdrant to be ready..."
	@for i in $$(seq 1 30); do \
		if curl -sf http://localhost:6333/collections > /dev/null 2>&1; then \
			echo "Qdrant is ready!"; \
			break; \
		fi; \
		if [ "$$i" = "30" ]; then \
			echo "Qdrant failed to start in time"; \
			docker compose -f tests/compose-qdrant.yml logs; \
			exit 1; \
		fi; \
		echo -n "."; \
		sleep 1; \
	done

stop-qdrant:
	docker compose -f tests/compose-qdrant.yml down

TEST ?= .
test:
	make start-qdrant
	uv run pytest $(TEST); \
	EXIT_CODE=$$?; \
	make stop-qdrant; \
	exit $$EXIT_CODE

test_watch:
	make start-qdrant; \
	uv run ptw $(TEST); \
	EXIT_CODE=$$?; \
	make stop-qdrant; \
	exit $$EXIT_CODE

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --relative --diff-filter=d main . | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langgraph
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	uv run ruff check .
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE)
	[ "$(PYTHON_FILES)" = "" ] || uv run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

type:
	mkdir -p $(MYPY_CACHE) && uv run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --select I --fix $(PYTHON_FILES)
