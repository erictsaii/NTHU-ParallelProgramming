#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

pthread_barrier_t barrier;
int cpu_cnt;
int** D;
int* D_malloc;
int V;
int E;

void input(const char* filename) {
    // open file
    FILE* file = fopen(filename, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);

    // initialize matrix
    // int src, dst, dis;
    D = (int**)malloc(V * sizeof(int*));
    D_malloc = (int*)malloc(V * V * sizeof(int));
    // for (int i = 0; i < V; ++i) D[i] = (int *)malloc(V * sizeof(int));
    for (int i = 0; i < V; ++i) D[i] = D_malloc + i * V;
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (i == j)
                D[i][j] = 0;
            else
                D[i][j] = 1073741823;
        }
    }
    int tmp[300];
    if (E >= 100) {
        int j = 0;
        for (; j < E; j += 100) {
            fread(tmp, sizeof(int), 300, file);
            for (int i = 0; i < 300; i += 3) {
                D[tmp[i]][tmp[i + 1]] = tmp[i + 2];
            }
        }
        for (int i = j - 100; i < E; ++i) {
            fread(tmp, sizeof(int), 3, file);
            D[tmp[0]][tmp[1]] = tmp[2];
        }
    } else {
        for (int i = 0; i < E; ++i) {
            fread(tmp, sizeof(int), 3, file);
            D[tmp[0]][tmp[1]] = tmp[2];
        }
    }

    fclose(file);
}

void* floyd_warshall(void* thread_id) {
    int id = *(int*)thread_id;
    for (int k = 0; k < V; ++k) {
        for (int i = id; i < V; i += cpu_cnt) {
            for (int j = 0; j < V; ++j) {
                if (D[i][j] > D[i][k] + D[k][j]) D[i][j] = D[i][k] + D[k][j];
            }
        }
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}

void output(const char* filename) {
    FILE* file = fopen(filename, "w");
    fwrite(D_malloc, sizeof(int), V * V, file);
    fclose(file);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    cpu_cnt = CPU_COUNT(&cpu_set);

    // get input
    input(argv[1]);

    /*initialize*/
    pthread_t threads[cpu_cnt];
    int ID[cpu_cnt];
    pthread_barrier_init(&barrier, NULL, cpu_cnt);

    for (int i = 0; i < cpu_cnt; ++i) {
        ID[i] = i;
        pthread_create(&threads[i], NULL, floyd_warshall, (void*)&ID[i]);
    }
    for (int i = 0; i < cpu_cnt; ++i) {
        pthread_join(threads[i], NULL);
    }

    // output
    output(argv[2]);
    pthread_exit(NULL);
}