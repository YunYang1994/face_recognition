#ifndef TEST_H
#define TEST_H
#include <stdio.h>
#define EPS .005
extern int tests_total;
extern int tests_fail;
#define TEST(EX) do { ++tests_total; if(!(EX)) {\
    fprintf(stderr, "failed: [%s] testing [%s] in %s, line %d\n", __FUNCTION__, #EX, __FILE__, __LINE__); \
    ++tests_fail; }} while (0)

void run_tests();
#endif
