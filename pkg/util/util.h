#ifndef BINDING_UTIL_H
#define BINDING_UTIL_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct charArray {
    unsigned long len;
    char **data;
} charArray;
struct charArray *makeCharArray(unsigned long len);
void setArrayString(const struct charArray *arr, char *s, unsigned long n);
void freeCharArray(struct charArray *arr);

#ifdef __cplusplus
}
#endif
#endif //BINDING_UTIL_H
