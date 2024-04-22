#include <cstdlib>
#include "util.h"

struct charArray * makeCharArray(unsigned long len) {
    auto arr = new(charArray);
    arr->len = len;
    arr->data = static_cast<char**>(calloc(sizeof(char*), len));
    return arr;
}

void setArrayString(const struct charArray * arr, char *s, unsigned long n) {
    arr->data[n] = s;
}

void freeCharArray(struct charArray *arr) {
    for (unsigned long i = 0; i < arr->len; i++)
        free(arr->data[i]);
    free(arr);
}