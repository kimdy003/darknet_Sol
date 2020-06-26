
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE
#include <unistd.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include "thpool.h"
#include <sched.h>

#ifdef THPOOL_DEBUG
#define THPOOL_DEBUG 1
#else
#define THPOOL_DEBUG 0
#endif

#if !defined(DISABLE_PRINT) || defined(THPOOL_DEBUG)
#define err(str) fprintf(stderr, str)
#else
#define err(str)
#endif

static volatile int threads_keepalive;
static volatile int threads_on_hold;

#if 0 // 2020 0206 hojin
/* ========================== STRUCTURES ============================ */


/* Binary semaphore */
typedef struct bsem {
	pthread_mutex_t mutex;
	pthread_cond_t   cond;
	int v;
} bsem;


/* Job */
typedef struct job{
	struct job*  prev;                   /* pointer to previous job   */
	void   (*function)(void* arg);       /* function pointer          */
	void*  arg;                          /* function's argument       */
} job;


/* Job queue */
typedef struct jobqueue{
	pthread_mutex_t rwmutex;             /* used for queue r/w access */
	job  *front;                         /* pointer to front of queue */
	job  *rear;                          /* pointer to rear  of queue */
	bsem *has_jobs;                      /* flag as binary semaphore  */
	int   len;                           /* number of jobs in queue   */
} jobqueue;


/* Thread */
typedef struct thread{
	int       id;                        /* friendly id               */
	pthread_t pthread;                   /* pointer to actual thread  */
	struct thpool_* thpool_p;            /* access to thpool          */
} thread;


/* Threadpool */
typedef struct thpool_{
	thread**   threads;                  /* pointer to threads        */
	volatile int num_threads_alive;      /* threads currently alive   */
	volatile int num_threads_working;    /* threads currently working */
	pthread_mutex_t  thcount_lock;       /* used for thread count etc */
	pthread_cond_t  threads_all_idle;    /* signal to thpool_wait     */
	jobqueue  jobqueue;                  /* job queue                 */
} thpool_;

#endif

/* ========================== PROTOTYPES ============================ */

static int thread_init(thpool_ *thpool_p, struct thread **thread_p, int id);
static void *thread_do(struct thread *thread_p);
static void thread_hold(int sig_id);
static void thread_destroy(struct thread *thread_p);

static int jobqueue_init(jobqueue *jobqueue_p);
static void jobqueue_clear(jobqueue *jobqueue_p);
static void jobqueue_push(jobqueue *jobqueue_p, struct job *newjob_p);
static struct job *jobqueue_pull(jobqueue *jobqueue_p);
static void jobqueue_destroy(jobqueue *jobqueue_p);

static void bsem_init(struct bsem *bsem_p, int value);
static void bsem_reset(struct bsem *bsem_p);
static void bsem_post(struct bsem *bsem_p);
static void bsem_post_all(struct bsem *bsem_p);
static void bsem_wait(struct bsem *bsem_p);

/* ========================== THREADPOOL ============================ */

/* Initialise thread pool */
struct thpool_ *thpool_init(int num_threads)
{

	threads_on_hold = 0;
	threads_keepalive = 1;

	if (num_threads < 0)
	{
		num_threads = 0;
	}

	/* Make new thread pool */
	thpool_ *thpool_p;
	thpool_p = (struct thpool_ *)malloc(sizeof(struct thpool_));
	if (thpool_p == NULL)
	{
		err("thpool_init(): Could not allocate memory for thread pool\n");
		return NULL;
	}
	thpool_p->num_threads_alive = 0;
	thpool_p->num_threads_working = 0;
#ifdef PRIORITY
	thpool_p->pri = NULL;
#endif

	/* Initialise the job queue */
	if (jobqueue_init(&thpool_p->jobqueue) == -1)
	{
		err("thpool_init(): Could not allocate memory for job queue\n");
		free(thpool_p);
		return NULL;
	}

	/* Make threads in pool */
	thpool_p->threads = (struct thread **)malloc(num_threads * sizeof(struct thread *));
	if (thpool_p->threads == NULL)
	{
		err("thpool_init(): Could not allocate memory for threads\n");
		jobqueue_destroy(&thpool_p->jobqueue);
		free(thpool_p);
		return NULL;
	}

	pthread_mutex_init(&(thpool_p->thcount_lock), NULL);
	pthread_cond_init(&thpool_p->threads_all_idle, NULL);

	/* Thread init */
	int n;
	cpu_set_t cpuset;

	for (n = 0; n < num_threads; n++)
	{
		thread_init(thpool_p, &thpool_p->threads[n], n);

#ifdef CPU
		if (n == (num_threads - 1))
		{
#endif
			thpool_p->threads[n]->flag = 1;
#ifdef CPU
		}
		else
		{
			thpool_p->threads[n]->flag = 0;
		}
#endif

		/* kmsjames 2020 0215 bug fix for pinning each thread on a specified CPU */
		CPU_ZERO(&cpuset);
		CPU_SET(n % 8, &cpuset); //only this thread has the affinity for the 'n'-th CPU
		pthread_setaffinity_np(thpool_p->threads[n]->pthread, sizeof(cpu_set_t), &cpuset);

#if THPOOL_DEBUG
		printf("THPOOL_DEBUG: Created thread %d in pool \n", n);
#endif
	}

	/* Wait for threads to initialize */
	while (thpool_p->num_threads_alive != num_threads)
	{
	}
	return thpool_p;
}

/* Add work to the thread pool */
int thpool_add_work(thpool_ *thpool_p, void (*function_p)(void *), void *arg_p)
{
	job *newjob;

	newjob = (struct job *)malloc(sizeof(struct job));
	if (newjob == NULL)
	{
		err("thpool_add_work(): Could not allocate memory for new job\n");
		return -1;
	}

	/* add function and argument */
	newjob->function = function_p;
	newjob->arg = arg_p;
	newjob->flag = 1;
	thpool_p->pri = ((th_arg *)arg_p)->pri;

	/* add job to queue */
	jobqueue_push(&thpool_p->jobqueue, newjob);
	return 0;
}

/* Wait until all jobs have finished */
void thpool_wait(thpool_ *thpool_p)
{
	pthread_mutex_lock(&thpool_p->thcount_lock);
	while (thpool_p->jobqueue.len || thpool_p->num_threads_working)
	{
		pthread_cond_wait(&thpool_p->threads_all_idle, &thpool_p->thcount_lock);
	}
	pthread_mutex_unlock(&thpool_p->thcount_lock);
}

/* Destroy the threadpool */
void thpool_destroy(thpool_ *thpool_p)
{
	/* No need to destory if it's NULL */
	if (thpool_p == NULL)
		return;

	volatile int threads_total = thpool_p->num_threads_alive;

	/* End each thread 's infinite loop */
	threads_keepalive = 0;

	/* Give one second to kill idle threads*/
	double TIMEOUT = 1.0;
	time_t start, end;
	double tpassed = 0.0;
	time(&start);
	while (tpassed < TIMEOUT && thpool_p->num_threads_alive)
	{
		bsem_post_all(thpool_p->jobqueue.has_jobs);
		time(&end);
		tpassed = difftime(end, start);
	}

	/* Poll remaining threads */
	while (thpool_p->num_threads_alive)
	{
		fprintf(stderr, "xxxxxxxxxxxxxxx\n");
		bsem_post_all(thpool_p->jobqueue.has_jobs);
		sleep(1);
	}

	/* Job queue cleanup */
	jobqueue_destroy(&thpool_p->jobqueue);
	/* Deallocs */
	int n;
	for (n = 0; n < threads_total; n++)
	{
		thread_destroy(thpool_p->threads[n]);
	}
	free(thpool_p->threads);
	free(thpool_p);
}

/* Pause all threads in threadpool */
void thpool_pause(thpool_ *thpool_p)
{
	int n;
	for (n = 0; n < thpool_p->num_threads_alive; n++)
	{
		pthread_kill(thpool_p->threads[n]->pthread, SIGUSR1);
	}
}

/* Resume all threads in threadpool */
void thpool_resume(thpool_ *thpool_p)
{
	// resuming a single threadpool hasn't been
	// implemented yet, meanwhile this supresses
	// the warnings
	(void)thpool_p;

	threads_on_hold = 0;
}

int thpool_num_threads_working(thpool_ *thpool_p)
{
	return thpool_p->num_threads_working;
}

/* ============================ THREAD ============================== */

/* Initialize a thread in the thread pool
 *
 * @param thread        address to the pointer of the thread to be created
 * @param id            id to be given to the thread
 * @return 0 on success, -1 otherwise.
 */
static int thread_init(thpool_ *thpool_p, struct thread **thread_p, int id)
{

	*thread_p = (struct thread *)malloc(sizeof(struct thread));
	if (*thread_p == NULL)
	{
		err("thread_init(): Could not allocate memory for thread\n");
		return -1;
	}

	(*thread_p)->thpool_p = thpool_p;
	(*thread_p)->id = id;

	pthread_create(&(*thread_p)->pthread, NULL, (void *)thread_do, (*thread_p));
	pthread_detach((*thread_p)->pthread);
	return 0;
}

/* Sets the calling thread on hold */
static void thread_hold(int sig_id)
{
	(void)sig_id;
	threads_on_hold = 1;
	while (threads_on_hold)
	{
		sleep(1);
	}
}

/* What each thread is doing
*
* In principle this is an endless loop. The only time this loop gets interuppted is once
* thpool_destroy() is invoked or the program exits.
*
* @param  thread        thread that will run this function
* @return nothing
*/
static void *thread_do(struct thread *thread_p)
{

	/* Set thread name for profiling and debuging */
	char thread_name[128] = {0};
	sprintf(thread_name, "thread-pool-%d", thread_p->id);

#if defined(__linux__)
	/* Use prctl instead to prevent using _GNU_SOURCE flag and implicit declaration */
	prctl(PR_SET_NAME, thread_name);
#elif defined(__APPLE__) && defined(__MACH__)
	pthread_setname_np(thread_name);
#else
	err("thread_do(): pthread_setname_np is not supported on this system");
#endif

	/* Assure all threads have been created before starting serving */
	thpool_ *thpool_p = thread_p->thpool_p;

	/* Register signal handler */
	struct sigaction act;
	sigemptyset(&act.sa_mask);
	act.sa_flags = 0;
	act.sa_handler = thread_hold;
	if (sigaction(SIGUSR1, &act, NULL) == -1)
	{
		err("thread_do(): cannot handle SIGUSR1");
	}

	/* Mark thread as alive (initialized) */
	pthread_mutex_lock(&thpool_p->thcount_lock);
	thpool_p->num_threads_alive += 1;
	pthread_mutex_unlock(&thpool_p->thcount_lock);

	while (threads_keepalive)
	{

#if 0
		// doyoung
		if (thread_p->id == 0 && thpool_p->jobqueue.len == 0)
		{
			continue;
		}
		if (thpool_p->jobqueue.front != NULL)
		{
			if (thread_p->id == 0 && ((th_arg *)thpool_p->jobqueue.front->arg)->type == 0)
			{
				//fprintf(stderr, " [%d - %d]  id = 0 && conv \n", ((th_arg*)thpool_p->jobqueue.front->arg)->id, ((th_arg*)thpool_p->jobqueue.front->arg)->n);
				continue;
			}
		}
#endif

		bsem_wait(thpool_p->jobqueue.has_jobs);

		if (threads_keepalive)
		{
			pthread_mutex_lock(&thpool_p->thcount_lock);
			thpool_p->num_threads_working++;
			pthread_mutex_unlock(&thpool_p->thcount_lock);

			/* Read job from queue and execute it */
			//2020 0311 doyoung
			#ifdef STREAM
				void (*func_buff)(void *, int);
				void *arg_buff;
				job *job_p = jobqueue_pull(&thpool_p->jobqueue);
				if (job_p)
				{
					func_buff = job_p->function;
					arg_buff = job_p->arg;
					((th_arg *)arg_buff)->flag = thread_p->flag;
					func_buff(arg_buff, thread_p->id);
					free(job_p);
				}
			#else
				void (*func_buff)(void *);
				void *arg_buff;
				job *job_p = jobqueue_pull(&thpool_p->jobqueue);
				if (job_p)
				{
					func_buff = job_p->function;
					arg_buff = job_p->arg;
					((th_arg *)arg_buff)->flag = thread_p->flag;
					func_buff(arg_buff);
					free(job_p);
				}
			#endif
			pthread_mutex_lock(&thpool_p->thcount_lock);
			thpool_p->num_threads_working--;
			if (!thpool_p->num_threads_working)
			{
				pthread_cond_signal(&thpool_p->threads_all_idle);
			}
			pthread_mutex_unlock(&thpool_p->thcount_lock);
		}
	}
	pthread_mutex_lock(&thpool_p->thcount_lock);
	thpool_p->num_threads_alive--;
	pthread_mutex_unlock(&thpool_p->thcount_lock);

	return NULL;
}

/* Frees a thread  */
static void thread_destroy(thread *thread_p)
{
	free(thread_p);
}

/* ============================ JOB QUEUE =========================== */

/* Initialize queue */
static int jobqueue_init(jobqueue *jobqueue_p)
{
	jobqueue_p->len = 0;
	jobqueue_p->front = NULL;
	jobqueue_p->rear = NULL;

	jobqueue_p->has_jobs = (struct bsem *)malloc(sizeof(struct bsem));
	if (jobqueue_p->has_jobs == NULL)
	{
		return -1;
	}

	pthread_mutex_init(&(jobqueue_p->rwmutex), NULL);
	bsem_init(jobqueue_p->has_jobs, 0);

	return 0;
}

/* Clear the queue */
static void jobqueue_clear(jobqueue *jobqueue_p)
{

	while (jobqueue_p->len)
	{
		free(jobqueue_pull(jobqueue_p));
	}

	jobqueue_p->front = NULL;
	jobqueue_p->rear = NULL;
	bsem_reset(jobqueue_p->has_jobs);
	jobqueue_p->len = 0;
}

/* Add (allocated) job to queue */
static void jobqueue_push(jobqueue *jobqueue_p, struct job *newjob)
{

	pthread_mutex_lock(&jobqueue_p->rwmutex);
	#ifdef PRIORITY
		char *my_pri;
		my_pri = ((th_arg *)newjob->arg)->pri;
	#endif
	newjob->prev = NULL;

	#ifdef PRIORITY
		if(strcmp(my_pri, "h") == 0){
			if(jobqueue_p->H_tail == NULL){
				if(jobqueue_p->len == 0){
					jobqueue_p->front = newjob;
					jobqueue_p->H_tail = newjob;
				}
				else{
					newjob->prev = jobqueue_p->front;
					jobqueue_p->front = newjob;
					jobqueue_p->H_tail = newjob;
				}
			}
			else{
				newjob->prev = jobqueue_p->H_tail->prev;
				jobqueue_p->H_tail->prev = newjob;
				jobqueue_p->H_tail = newjob;
			}
		}
		else if(strcmp(my_pri, "m") == 0){
			if(jobqueue_p->M_tail == NULL){
				if(jobqueue_p->len == 0){
					jobqueue_p->front = newjob;
					jobqueue_p->M_tail = newjob;
				}
				else{
					if(jobqueue_p->H_tail != NULL && jobqueue_p->L_tail == NULL){
						jobqueue_p->H_tail->prev = newjob;
						jobqueue_p->M_tail = newjob;
					}
					else if(jobqueue_p->H_tail == NULL && jobqueue_p->L_tail != NULL){
						newjob->prev = jobqueue_p->front;
						jobqueue_p->front = newjob;
						jobqueue_p->M_tail = newjob;
					}
					else{
						newjob->prev = jobqueue_p->H_tail->prev;
						jobqueue_p->H_tail->prev = newjob;
						jobqueue_p->M_tail = newjob;
					}
				}
			}
			else{
				newjob->prev = jobqueue_p->M_tail->prev;
				jobqueue_p->M_tail->prev = newjob;
				jobqueue_p->M_tail = newjob;
			}

		}
		else if(strcmp(my_pri, "l") == 0){
			if(jobqueue_p->L_tail == NULL){
				if(jobqueue_p->len == 0){
					jobqueue_p->front = newjob;
					jobqueue_p->L_tail = newjob;
				}
				else{
					if(jobqueue_p->M_tail != NULL){
						jobqueue_p->M_tail->prev = newjob;
						jobqueue_p->L_tail = newjob;
					}
					else if(jobqueue_p->H_tail != NULL){
						jobqueue_p->H_tail->prev = newjob;
						jobqueue_p->L_tail = newjob;
					}
				}
			}
			else{
				jobqueue_p->L_tail->prev = newjob;
				jobqueue_p->L_tail = newjob;
			}
		}
		else{
			return;
		}

	#else
		switch (jobqueue_p->len)
		{

		case 0: /* if no jobs in queue */
			jobqueue_p->front = newjob;
			jobqueue_p->rear = newjob;
			break;

		default: /* if jobs in queue */
			jobqueue_p->rear->prev = newjob;
			jobqueue_p->rear = newjob;
		}
	#endif
	jobqueue_p->len++;

	bsem_post(jobqueue_p->has_jobs);
	pthread_mutex_unlock(&jobqueue_p->rwmutex);
}

/* Get first job from queue(removes it from queue) */
static struct job *jobqueue_pull(jobqueue *jobqueue_p)
{

	pthread_mutex_lock(&jobqueue_p->rwmutex);
	job *job_p = jobqueue_p->front;
	#ifdef PRIORITY
		char *my_pri;
		if(job_p != NULL)
		    my_pri = ((th_arg*)job_p->arg)->pri;

		//fprintf(stderr, "jobqueue len : %d \n", jobqueue_p->len);
		//fprintf(stderr, "jobqueue pri : %s \n", my_pri);
	
		switch (jobqueue_p->len)
		{

		case 0: /* if no jobs in queue */
			break;

		case 1: /* if one job in queue */
			if(strcmp(my_pri, "h") == 0){
				jobqueue_p->front = NULL;
				jobqueue_p->H_tail = NULL;
				jobqueue_p->len = 0;
			}
			else if(strcmp(my_pri, "m") == 0){
				jobqueue_p->front = NULL;
				jobqueue_p->M_tail = NULL;
				jobqueue_p->len = 0;
			}
			else if(strcmp(my_pri, "l") == 0){
				jobqueue_p->front = NULL;
				jobqueue_p->L_tail = NULL;
				jobqueue_p->len = 0;
			}
			else{
			    break;
			}
			break;

		default: /* if >1 jobs in queue */
			if(jobqueue_p->front == jobqueue_p->H_tail){
				jobqueue_p->H_tail = NULL;
			}
			else if(jobqueue_p->front == jobqueue_p->M_tail){
				jobqueue_p->M_tail = NULL;
			}
			jobqueue_p->front = job_p->prev;
			jobqueue_p->len--;
#if 0
			if(strcmp(my_pri, "H") == 0){
				if(jobqueue_p->front == jobqueue_p->H_tail){
					jobqueue_p->H_tail = NULL;
				}
				jobqueue_p->front = job_p->prev;
				jobqueue_p->len--;
			}
			else if(strcmp(my_pri, "M") == 0){
				if(jobqueue_p->front == jobqueue_p->M_tail){
					jobqueue_p->M_tail = NULL;
				}
				jobqueue_p->front = job_p->prev;
				jobqueue_p->len--;
			}
			else if(strcmp(my_pri, "L") == 0){
				jobqueue_p->front = job_p->prev;
				jobqueue_p->len--;
			}

#endif			/* more than one job in queue -> post it */
			bsem_post(jobqueue_p->has_jobs);
		}
	#else
		switch (jobqueue_p->len)
		{

		case 0: /* if no jobs in queue */
			break;

		case 1: /* if one job in queue */
			jobqueue_p->front = NULL;
			jobqueue_p->rear = NULL;
			jobqueue_p->len = 0;
			break;

		default: /* if >1 jobs in queue */
			jobqueue_p->front = job_p->prev;
			jobqueue_p->len--;
			/* more than one job in queue -> post it */
			bsem_post(jobqueue_p->has_jobs);
		}
	#endif

	pthread_mutex_unlock(&jobqueue_p->rwmutex);
	return job_p;
}

/* Free all queue resources back to the system */
static void jobqueue_destroy(jobqueue *jobqueue_p)
{
	jobqueue_clear(jobqueue_p);
	free(jobqueue_p->has_jobs);
}

/* ======================== SYNCHRONISATION ========================= */

/* Init semaphore to 1 or 0 */
static void bsem_init(bsem *bsem_p, int value)
{
	if (value < 0 || value > 1)
	{
		err("bsem_init(): Binary semaphore can take only values 1 or 0");
		exit(1);
	}
	pthread_mutex_init(&(bsem_p->mutex), NULL);
	pthread_cond_init(&(bsem_p->cond), NULL);
	bsem_p->v = value;
}

/* Reset semaphore to 0 */
static void bsem_reset(bsem *bsem_p)
{
	bsem_init(bsem_p, 0);
}

/* Post to at least one thread */
static void bsem_post(bsem *bsem_p)
{
	pthread_mutex_lock(&bsem_p->mutex);
	bsem_p->v = 1;
	pthread_cond_signal(&bsem_p->cond);
	pthread_mutex_unlock(&bsem_p->mutex);
}

/* Post to all threads */
static void bsem_post_all(bsem *bsem_p)
{
	pthread_mutex_lock(&bsem_p->mutex);
	bsem_p->v = 1;
	pthread_cond_broadcast(&bsem_p->cond);
	pthread_mutex_unlock(&bsem_p->mutex);
}

/* Wait on semaphore until semaphore has value 0 */
static void bsem_wait(bsem *bsem_p)
{
	pthread_mutex_lock(&bsem_p->mutex);
	while (bsem_p->v != 1)
	{
		pthread_cond_wait(&bsem_p->cond, &bsem_p->mutex);
	}
	bsem_p->v = 0;
	pthread_mutex_unlock(&bsem_p->mutex);
}
