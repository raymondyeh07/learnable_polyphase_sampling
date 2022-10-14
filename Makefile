.PHONY: test
test:
	$(MAKE) -C learn_poly_sampling test
	$(MAKE) -C learn_poly_sampling sanity-checks-nogpu
