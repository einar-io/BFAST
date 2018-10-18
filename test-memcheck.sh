#!/usr/bin/env bash
futhark-test --compiler=/bin/true --runner=./memcheck-runner.sh src/tests/tests.fut
