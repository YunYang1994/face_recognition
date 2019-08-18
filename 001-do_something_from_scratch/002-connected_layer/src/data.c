#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "uwnet.h"
#include "list.h"

data random_batch(data d, int n)
{
    matrix X = {0};
    matrix y = {0};
    X.rows = y.rows = n;
    X.cols = d.X.cols;
    y.cols = d.y.cols;
    X.data = calloc(n*X.cols, sizeof(float*));
    y.data = calloc(n*y.cols, sizeof(float*));
    int i, j;
    for(i = 0; i < n; ++i){
        int ind = rand()%d.X.rows;
        for(j = 0; j < X.cols; ++j){
            X.data[i*X.cols + j] = d.X.data[ind*X.cols + j];
        }
        for(j = 0; j < y.cols; ++j){
            y.data[i*y.cols + j] = d.y.data[ind*y.cols + j];
        }
    }
    data c;
    c.X = X;
    c.y = y;
    return c;
}

list *get_lines(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) {
        fprintf(stderr, "Couldn't open file %s\n", filename);
        exit(0);
    }
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

data load_image_classification_data(char *images, char *label_file)
{
    list *image_list = get_lines(images);
    list *label_list = get_lines(label_file);
    int k = label_list->size;
    char **labels = (char **)list_to_array(label_list);

    int n = image_list->size;
    node *nd = image_list->front;
    int cols = 0;
    int i;
    int count = 0;
    matrix X;
    matrix y = make_matrix(n, k);
    while(nd){
        char *path = (char *)nd->val;
        image im = load_image(path);
        if (!cols) {
            cols = im.w*im.h*im.c;
            X = make_matrix(n, cols);
        }
        for (i = 0; i < cols; ++i){
            X.data[count*X.cols + i] = im.data[i];
        }

        for (i = 0; i < k; ++i){
            if(strstr(path, labels[i])){
                y.data[count*y.cols + i] = 1;
            }
        }
        ++count;
        nd = nd->next;
    }
    free_list(image_list);
    data d;
    d.X = X;
    d.y = y;
    return d;
}


char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    size_t size = 512;
    char *line = malloc(size*sizeof(char));
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while((line[curr-1] != '\n') && !feof(fp)){
        if(curr == size-1){
            size *= 2;
            line = realloc(line, size*sizeof(char));
            if(!line) {
                fprintf(stderr, "malloc failed %ld\n", size);
                exit(0);
            }
        }
        size_t readsize = size-curr;
        if(readsize > INT_MAX) readsize = INT_MAX-1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if(line[curr-1] == '\n') line[curr-1] = '\0';

    return line;
}

void free_data(data d)
{
    free_matrix(d.X);
    free_matrix(d.y);
}



