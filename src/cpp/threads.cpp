#include "threads.h"
#ifdef USE_PTHREADS
#include <pthread.h>
#endif
#include <openacc.h>

int numthreads = 1;
#ifdef USE_PTHREADS
pthread_mutex_t mmutex;
pthread_t p_thread[MAXTHREADS];
#endif
memshard memshards[MEMSHARDS];

void init_mutex() {
  #ifdef USE_PTHREADS
  pthread_mutex_init(&mmutex, NULL);
  #endif
  
  // Initialize GPU device
  #pragma acc init
}

void get_global_lock() {
  #ifdef USE_PTHREADS
  pthread_mutex_lock(&mmutex);
  #endif
}

void release_global_lock() {
  #ifdef USE_PTHREADS
  pthread_mutex_unlock(&mmutex);
  #endif
}

#ifdef USE_PTHREADS
void spawn_thread(int i, THREAD_RETURN_TYPE(THREAD_DECLARATOR *p)(void *), void *o) {
  pthread_create(&(p_thread[i]), NULL, p, o);
}

void join_thread(int i) {
  pthread_join(p_thread[i], 0);
}
#endif

void init_threads() {
  #ifdef USE_PTHREADS
  init_mutex();
  for (int i = 0; i < MEMSHARDS; i++)
    pthread_mutex_init(&(memshards[i].mutex), NULL);
  #endif
  
  // Initialize GPU for multi-threaded usage
  #pragma acc init
  
  // Set number of GPU gangs/workers based on hardware
  int num_devices = acc_get_num_devices(acc_device_default);
  if (num_devices > 0) {
    acc_set_device_num(0, acc_device_default);
    
    // Query GPU properties and adjust thread count accordingly
    int gpu_cores = acc_get_property(0, acc_device_default, acc_property_multiprocessors);
    int gpu_threads = acc_get_property(0, acc_device_default, acc_property_threads);
    
    if (gpu_cores > 0 && gpu_threads > 0) {
      // Adjust numthreads based on GPU capabilities if needed
      // This is a heuristic - you may want to tune this
      int gpu_parallel = min(gpu_cores * 32, MAXTHREADS);
      if (numthreads > gpu_parallel) {
        numthreads = gpu_parallel;
      }
    }
  }
}
