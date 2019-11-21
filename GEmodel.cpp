#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <exception>
// #include <boost/thread.hpp>
#include <thread>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
// Sun CC doesn't handle boost::iterator_adaptor yet
#if !defined(__SUNPRO_CC) || (__SUNPRO_CC > 0x530)
#include <boost/generator_iterator.hpp>
#endif


#define MAX_STRING 100
#define SIGMOID_BOUND 6
#define NEG_SAMPLING_POWER 0.75

const int hash_table_size = 30000000;
const int neg_table_size = 1e8;
const int sigmoid_table_size = 1000;

typedef float real;                    // Precision of float numbers

struct ClassVertex {
	double degree;
	char *name;
};

char poi_file[MAX_STRING], net_poi[MAX_STRING], net_poi_reg[MAX_STRING], net_poi_time[MAX_STRING], net_poi_word[MAX_STRING];
char emb_poi[MAX_STRING], emb_reg[MAX_STRING], emb_time[MAX_STRING], emb_word[MAX_STRING];
struct ClassVertex *vertex_poi;
struct ClassVertex *vertex_v;
struct ClassVertex *vertex_r;
struct ClassVertex *vertex_t;
struct ClassVertex *vertex_w;
int is_binary = 0, num_threads = 2, dim = 20, num_negative = 5;
int *vertex_hash_table, *word_hash_table, *region_hash_table, *time_hash_table, *neg_table_v, *neg_table_r, *neg_table_t, *neg_table_w ;
int max_num_vertices = 1000, num_vertices_poi = 0, num_vertices_v = 0, num_vertices_r = 0, num_vertices_t = 0, num_vertices_w = 0;
long long total_samples = 100, current_sample_count = 0, num_edges_vv = 0, num_edges_vr = 0, num_edges_vt = 0, num_edges_vw = 0;
real init_rho = 0.025, rho;
real *emb_vertex_v, *emb_vertex_r, *emb_vertex_t, *emb_vertex_w, *sigmoid_table;

int *vv_edge_source_id, *vv_edge_target_id, *vr_edge_source_id, *vr_edge_target_id, *vt_edge_source_id, *vt_edge_target_id, *vw_edge_source_id, *vw_edge_target_id;
double *vv_edge_weight, *vr_edge_weight, *vt_edge_weight, *vw_edge_weight;

// Parameters for edge sampling
long long *alias_vv, *alias_vr, *alias_vt, *alias_vw;
double *prob_vv, *prob_vr, *prob_vt, *prob_vw;

//random generator 
typedef boost::minstd_rand base_generator_type;
base_generator_type generator(42u);
boost::uniform_real<> uni_dist(0, 1);
boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(generator, uni_dist);



/* Build a hash table, mapping each vertex name to a unique vertex id */
unsigned int Hash(char *key)
{
	unsigned int seed = 131;
	unsigned int hash = 0;
	while (*key)
	{
		hash = hash * seed + (*key++);
	}
	return hash % hash_table_size;
}

void InitHashTable()
{
	vertex_hash_table = (int *)malloc(hash_table_size * sizeof(int));
	for (int k = 0; k != hash_table_size; k++) vertex_hash_table[k] = -1;

	region_hash_table = (int *)malloc(hash_table_size * sizeof(int));
	for (int k = 0; k != hash_table_size; k++) region_hash_table[k] = -1;

	time_hash_table = (int *)malloc(hash_table_size * sizeof(int));
	for (int k = 0; k != hash_table_size; k++) time_hash_table[k] = -1;

	word_hash_table = (int *)malloc(hash_table_size * sizeof(int));
	for (int k = 0; k != hash_table_size; k++) word_hash_table[k] = -1;
}

void InsertHashTable(char *key, int value, int flag)
{
	int addr = Hash(key);
	if (flag==0) {
		while (vertex_hash_table[addr] != -1) {
			addr = (addr + 1) % hash_table_size;
		}
		vertex_hash_table[addr] = value;
	}
	else if(flag==1){
		while (word_hash_table[addr] != -1) {
			addr = (addr + 1) % hash_table_size;
		}
		word_hash_table[addr] = value;
	}
	else if(flag==2){
		while (region_hash_table[addr] != -1) {
			addr = (addr + 1) % hash_table_size;
		}
		region_hash_table[addr] = value;
	}
	else{
		while (time_hash_table[addr] != -1) {
			addr = (addr + 1) % hash_table_size;
		}
		time_hash_table[addr] = value;
	}
}

int SearchHashTable(char *key, ClassVertex *vertex, int flag)
{
	int addr = Hash(key);
	//std::cout<<key<<"\n";
	if(flag==0){
		while (1)
		{
			if (vertex_hash_table[addr] == -1) return -1;
			if (!strcmp(key, vertex[vertex_hash_table[addr]].name)) return vertex_hash_table[addr];
			addr = (addr + 1) % hash_table_size;
		}
	}
	else if(flag==1){
		while (1)
		{
			if (word_hash_table[addr] == -1) return -1;
			if (!strcmp(key, vertex[word_hash_table[addr]].name)) return word_hash_table[addr];
			addr = (addr + 1) % hash_table_size;
		}
	}
	else if(flag==2){
		while (1)
		{
			if (region_hash_table[addr] == -1) return -1;
			if (!strcmp(key, vertex[region_hash_table[addr]].name)) return region_hash_table[addr];
			addr = (addr + 1) % hash_table_size;
		}
	}
	else{
		while (1)
		{
			if (time_hash_table[addr] == -1) return -1;
			if (!strcmp(key, vertex[time_hash_table[addr]].name)) return time_hash_table[addr];
			addr = (addr + 1) % hash_table_size;
		}
	}
	
	return -1;
}

/* Add a vertex to the vertex set */
int AddVertex(char *name, ClassVertex *&vertex, int &num_vertices, int flag)
{
	int length = strlen(name) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	vertex[num_vertices].name = (char *)calloc(length, sizeof(char));
	strcpy(vertex[num_vertices].name, name);
	vertex[num_vertices].degree = 0;
	num_vertices++;
	if (num_vertices + 2 >= max_num_vertices)
	{
		max_num_vertices += 1000;
		vertex = (struct ClassVertex *)realloc(vertex, max_num_vertices * sizeof(struct ClassVertex));
	}
	InsertHashTable(name, num_vertices - 1, flag);
	return num_vertices - 1;
}


/* Read network from the training file */
void ReadFile(char *network_file, long long &num_edges, int &num_vertices, 
	          int *&edge_source_id, int *&edge_target_id, double *&edge_weight, ClassVertex *&vertex, int hash_flag, int flag)
{
	FILE *fin;
	char name_v1[MAX_STRING], name_v2[MAX_STRING], str[2 * MAX_STRING + 10000];
	int vid;
	double weight;

	fin = fopen(network_file, "rb");
	if (fin == NULL)
	{
		printf("ERROR: network file not found!\n");
		exit(1);
	}
	num_edges = 0;
	while (fgets(str, sizeof(str), fin)) num_edges++;
	fclose(fin);
	//printf("Number of edges: %lld          \n", num_edges);

	edge_source_id = (int *)malloc(num_edges*sizeof(int));
	edge_target_id = (int *)malloc(num_edges*sizeof(int));
	edge_weight = (double *)malloc(num_edges*sizeof(double));
	if (edge_source_id == NULL || edge_target_id == NULL || edge_weight == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	fin = fopen(network_file, "rb");

	num_vertices = 0;
	for (int k = 0; k != num_edges; k++)
	{
		fscanf(fin, "%s %s %lf", name_v1, name_v2, &weight);

		/*if (k % 10000 == 0)
		{
			printf("Reading edges: %.3lf%%%c", k / (double)(num_edges + 1) * 100, 13);
			fflush(stdout);
		}*/
		if (flag==1){
			vid = SearchHashTable(name_v1, vertex, 0);
			if (vid == -1) printf("Error: false point type!\n");
			if (vertex[vid].degree == 0) {num_vertices++;}
			vertex[vid].degree += weight;
			edge_source_id[k] = vid;
		}
		else{
			vid = SearchHashTable(name_v1, vertex_poi, 0);
			edge_source_id[k] = vid;
		}
		
		
 		vid = SearchHashTable(name_v2, vertex, hash_flag);
		if (vid == -1) vid = AddVertex(name_v2, vertex, num_vertices, hash_flag);
		if (flag == 1 && vertex[vid].degree == 0) {num_vertices++;}
		vertex[vid].degree += weight;
		edge_target_id[k] = vid;

		edge_weight[k] = weight;
	}
	fclose(fin);
	//printf("Number of vertices: %lld          \n", num_vertices);
}

void ReadPOIs(char *POI_file){
	FILE *fin;
	char name[MAX_STRING], str[MAX_STRING+10];
	int num_poi = 0, vid;

	fin = fopen(POI_file, "rb");
	if (fin == NULL)
	{
		printf("ERROR: network file not found!\n");
		exit(1);
	}

	while (fgets(str, sizeof(str), fin)) num_poi++;
	fclose(fin);

	fin = fopen(POI_file, "rb");
	for (int k = 0; k != num_poi; k++)
	{
		fscanf(fin, "%s", name);
		vid = SearchHashTable(name, vertex_poi, 0);
		if (vid == -1) {
			vid = AddVertex(name, vertex_poi, num_vertices_poi, 0);
		}
	}
	fclose(fin);
}

void ReadData(){
	char *name;
	int max_num = 1000;
	/* Init vertex_v* 's v poit in different graph */
	for(int i=0; i<num_vertices_poi; i++){
		if (i + 2 >= max_num)
		{
			max_num += 1000;
			vertex_v = (struct ClassVertex *)realloc(vertex_v, max_num * sizeof(struct ClassVertex));
		}
		name = vertex_poi[i].name;
		//std::cout<<name<<"\n";
		int length = strlen(name) + 1;
		if (length > MAX_STRING) length = MAX_STRING;
		vertex_v[i].name = (char *)calloc(length, sizeof(char));
		strcpy(vertex_v[i].name, name);
		vertex_v[i].degree = 0;
	}

	ReadFile(net_poi, num_edges_vv, num_vertices_v, vv_edge_source_id, vv_edge_target_id, vv_edge_weight, vertex_v, 0, 1);
	std::cout<<"Number of edges in net_POI:"<<"\t";
	std::cout<<num_edges_vv<<"\n";
	std::cout<<"Number of vertices of POIs:"<<"\t";
	std::cout<<num_vertices_v<<"\n";

	max_num_vertices = 1000;
	ReadFile(net_poi_reg, num_edges_vr, num_vertices_r, vr_edge_source_id, vr_edge_target_id, vr_edge_weight, vertex_r, 2, 0);
	std::cout<<"Number of edges in net_POI_reg:"<<"\t";
	std::cout<<num_edges_vr<<"\n";
	std::cout<<"Number of vertices of regions:"<<"\t";
	std::cout<<num_vertices_r<<"\n";

	max_num_vertices = 1000;
	ReadFile(net_poi_time, num_edges_vt, num_vertices_t, vt_edge_source_id, vt_edge_target_id, vt_edge_weight, vertex_t, 3, 0);
	std::cout<<"Number of edges in net_POI_time:"<<"\t";
	std::cout<<num_edges_vt<<"\n";
	std::cout<<"Number of vertices of time slots:"<<"\t";
	std::cout<<num_vertices_t<<"\n";

	max_num_vertices = 1000;
	ReadFile(net_poi_word, num_edges_vw, num_vertices_w, vw_edge_source_id, vw_edge_target_id, vw_edge_weight, vertex_w, 1, 0);
	std::cout<<"Number of edges in net_POI_word:"<<"\t";
	std::cout<<num_edges_vw<<"\n";
	std::cout<<"Number of vertices of words:"<<"\t";
	std::cout<<num_vertices_w<<"\n";

	free(vertex_hash_table);
	free(word_hash_table);
	free(region_hash_table);
}

/* The alias sampling algorithm, which is used to sample an edge in O(1) time. */
void InitAliasTable(long long *&alias, double *&prob, long long num_edges, double *edge_weight)
{
	alias = (long long *)malloc(num_edges*sizeof(long long));
	prob = (double *)malloc(num_edges*sizeof(double));
	if (alias == NULL || prob == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double *norm_prob = (double*)malloc(num_edges*sizeof(double));
	long long *large_block = (long long*)malloc(num_edges*sizeof(long long));
	long long *small_block = (long long*)malloc(num_edges*sizeof(long long));
	if (norm_prob == NULL || large_block == NULL || small_block == NULL)
	{
		printf("Error: memory allocation failed!\n");
		exit(1);
	}

	double sum = 0;
	long long cur_small_block, cur_large_block;
	long long num_small_block = 0, num_large_block = 0;

	for (long long k = 0; k != num_edges; k++) sum += edge_weight[k];
	for (long long k = 0; k != num_edges; k++) norm_prob[k] = edge_weight[k] * num_edges / sum;

	for (long long k = num_edges - 1; k >= 0; k--)
	{
		if (norm_prob[k]<1)
			small_block[num_small_block++] = k;
		else
			large_block[num_large_block++] = k;
	}

	while (num_small_block && num_large_block)
	{
		cur_small_block = small_block[--num_small_block];
		cur_large_block = large_block[--num_large_block];
		prob[cur_small_block] = norm_prob[cur_small_block];
		alias[cur_small_block] = cur_large_block;
		norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
		if (norm_prob[cur_large_block] < 1)
			small_block[num_small_block++] = cur_large_block;
		else
			large_block[num_large_block++] = cur_large_block;
	}

	while (num_large_block) prob[large_block[--num_large_block]] = 1;
	while (num_small_block) prob[small_block[--num_small_block]] = 1;

	free(norm_prob);
	free(small_block);
	free(large_block);
}

void InitAlias()
{
	InitAliasTable(alias_vv, prob_vv, num_edges_vv, vv_edge_weight);
	InitAliasTable(alias_vr, prob_vr, num_edges_vr, vr_edge_weight);
	InitAliasTable(alias_vt, prob_vt, num_edges_vt, vt_edge_weight);
	InitAliasTable(alias_vw, prob_vw, num_edges_vw, vw_edge_weight);
}

long long SampleAnEdge(double rand_value1, double rand_value2, int num_edges, long long *alias, double *prob)
{
	long long k = (long long)num_edges * rand_value1;
	return rand_value2 < prob[k] ? k : alias[k];
}

/* Initialize the vertex embedding and the context embedding */
void InitVector()
{
	long long a, b;
	//int num;
	//vertex of poi
	// emb_vertex_v = (real *)memalign( 128, (long long)num_vertices_poi * dim * sizeof(real));
	posix_memalign((void **)&emb_vertex_v,128, (long long)num_vertices_poi * dim * sizeof(real));
	if (emb_vertex_v == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices_poi; a++)
		emb_vertex_v[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
		//emb_vertex_v[a * dim + b] = 0;

	//vertex of region
	// emb_vertex_r = (real *)memalign(128,(long long)num_vertices_r * dim * sizeof(real));
	posix_memalign((void **)&emb_vertex_r,128, (long long)num_vertices_r * dim * sizeof(real));
	if (emb_vertex_r == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices_r; a++)
		emb_vertex_r[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
		//emb_vertex_r[a * dim + b] = 0;

	//vertex of time
	// emb_vertex_t = (real *)memalign(128,(long long)num_vertices_t * dim * sizeof(real));
	posix_memalign((void **)&emb_vertex_t,128, (long long)num_vertices_t * dim * sizeof(real));
	if (emb_vertex_t == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices_t; a++)
		emb_vertex_t[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
		//emb_vertex_t[a * dim + b] = 0;

	//vertex of word
	// emb_vertex_w = (real *)memalign(128,(long long)num_vertices_w * dim * sizeof(real));
	posix_memalign((void **)&emb_vertex_w,128, (long long)num_vertices_w * dim * sizeof(real));
	if (emb_vertex_w == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
	for (b = 0; b < dim; b++) for (a = 0; a < num_vertices_w; a++)
		emb_vertex_w[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
		//emb_vertex_w[a * dim + b] = 0;
}

/* Sample negative vertex samples according to vertex degrees */
void InitNegTable(int *&neg_table, int num_vertices, ClassVertex *vertex)
{
	double sum = 0, cur_sum = 0, por = 0;
	int vid = 0;
	neg_table = (int *)malloc(neg_table_size * sizeof(int));
	
	for (int k = 0; k != num_vertices; k++) sum += pow(vertex[k].degree, NEG_SAMPLING_POWER);
	for (int k = 0; k != neg_table_size; k++)
	{
		if ((double)(k + 1) / neg_table_size > por)
		{
			cur_sum += pow(vertex[vid].degree, NEG_SAMPLING_POWER);
			por = cur_sum / sum;
			vid++;
		}
		neg_table[k] = vid - 1;
	}
}

void InitNeg()
{
	InitNegTable(neg_table_v, num_vertices_v, vertex_v);
	InitNegTable(neg_table_r, num_vertices_r, vertex_r);
	InitNegTable(neg_table_t, num_vertices_t, vertex_t);
	InitNegTable(neg_table_w, num_vertices_w, vertex_w);
}

/* Fastly compute sigmoid function */
void InitSigmoidTable()
{
	real x;
	sigmoid_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real));
	for (int k = 0; k != sigmoid_table_size; k++)
	{
		x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
		sigmoid_table[k] = 1 / (1 + exp(-x));
	}
}

real FastSigmoid(real x)
{
	if (x > SIGMOID_BOUND) return 1;
	else if (x < -SIGMOID_BOUND) return 0;
	int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
	return sigmoid_table[k];
}

/* Fastly generate a random integer */
int Rand(unsigned long long &seed)
{
	seed = seed * 25214903917 + 11;
	return (seed >> 16) % neg_table_size;
}

/* Update embeddings */
void Update(real *vec_u, real *vec_v, real *vec_error, int label)
{
	real x = 0, g;

	for (int c = 0; c != dim; c++) x += vec_u[c] * vec_v[c];
	if (isnan(x)){
		std::cout<<"错了"<<std::endl;
	}
	g = (label - FastSigmoid(x)) * rho;
	for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
	for (int c = 0; c != dim; c++) vec_v[c] += g * vec_u[c];
}

void *TrainLINEThread(long id)
{
	long long u, v, lu, lv, target, label;
	long long count = 0, last_count = 0, curedge;
	unsigned long long seed = (long long)id;
	int *neg_table, *edge_source_id, *edge_target_id;
	real *emb_vertex_target, *emb_context_target;
	real *vec_error = (real *)calloc(dim, sizeof(real));

	while (1)
	{
		//judge for exit
		if (count > total_samples / num_threads + 2) break;

		if (count - last_count>10000)
		{
			current_sample_count += count - last_count;
			last_count = count;
			//printf("%cRho: %f  Progress: %.3lf%%", 13, rho, (real)current_sample_count / (real)(total_samples + 1) * 100);
			//fflush(stdout);
			rho = init_rho * (1 - current_sample_count / (real)(total_samples + 1));
			if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
		}

		int a = count%4;
		//std::cout<<a<<"\n";

		
		switch(a){
		case 0:
			curedge = SampleAnEdge(uni(), uni(), num_edges_vr, alias_vr, prob_vr);
			neg_table = neg_table_r;
			emb_vertex_target = emb_vertex_r;
			edge_source_id = vr_edge_source_id;
			edge_target_id = vr_edge_target_id;
			break;
			
		case 1:
			curedge = SampleAnEdge(uni(), uni(), num_edges_vt, alias_vt, prob_vt);
			neg_table = neg_table_t;
			emb_vertex_target = emb_vertex_t;
			edge_source_id = vt_edge_source_id;
			edge_target_id = vt_edge_target_id;
			break;
		case 2:
			curedge = SampleAnEdge(uni(), uni(), num_edges_vw, alias_vw, prob_vw);
			neg_table = neg_table_w;
			emb_vertex_target = emb_vertex_w;
			edge_source_id = vw_edge_source_id;
			edge_target_id = vw_edge_target_id;
			break;
		default:
			curedge = SampleAnEdge(uni(), uni(), num_edges_vv, alias_vv, prob_vv);
			neg_table = neg_table_v;
			emb_vertex_target = emb_vertex_v;
			edge_source_id = vv_edge_source_id;
			edge_target_id = vv_edge_target_id;
			break;
		}

		u = edge_source_id[curedge];
		v = edge_target_id[curedge];

		lu = u * dim;
		for (int c = 0; c != dim; c++) vec_error[c] = 0;

		// NEGATIVE SAMPLING
		for (int d = 0; d != num_negative + 1; d++)
		{
			if (d == 0)
			{
				target = v;
				label = 1;
			}
			else
			{
				target = neg_table[Rand(seed)];
				label = 0;
			}
			lv = target * dim;

			Update(&emb_vertex_v[lu], &emb_vertex_target[lv], vec_error, label);

		}
		for (int c = 0; c != dim; c++) emb_vertex_v[c + lu] += vec_error[c];
		count++;
	}
	free(vec_error);
	return NULL;
}

void OutputFile(char emb_file[100], int num_vertices, ClassVertex *vertex, real *emb_vertex){
	FILE *fo = fopen(emb_file, "wb");
	std::cout<<"outputfile... "<<num_vertices<<"\n";
	// fprintf(fo, "%d %d\n", num_vertices, dim);
	for (int a = 0; a < num_vertices; a++)
	{
		//std::cout<<a<<"\n";
		//std::cout<<vertex[a].name<<"\n";
		fprintf(fo, "%s ", vertex[a].name);
		if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_vertex[a * dim + b], sizeof(real), 1, fo);
		else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_vertex[a * dim + b]);
		fprintf(fo, "\n");
	}
	fclose(fo);
}

void Output()
{
	OutputFile(emb_poi, num_vertices_poi, vertex_v, emb_vertex_v);
	OutputFile(emb_reg, num_vertices_r, vertex_r, emb_vertex_r);
	OutputFile(emb_time, num_vertices_t, vertex_t, emb_vertex_t);
	OutputFile(emb_word, num_vertices_w, vertex_w, emb_vertex_w);
}

void TrainLINE() {
	long a;
	// boost::thread *pt = new boost::thread[num_threads];
	std::thread *pt = new std::thread[num_threads];

	printf("--------------------------------\n");
	printf("Samples: %lldM\n", total_samples / 1000000);
	printf("Negative: %d\n", num_negative);
	printf("Dimension: %d\n", dim);
	printf("Initial rho: %lf\n", init_rho);
	printf("Thread: %d\n", num_threads);
	printf("--------------------------------\n");

	InitHashTable();
	ReadPOIs(poi_file);
	ReadData();
	InitAlias();
	
	InitVector();
	InitNeg();
	InitSigmoidTable();

	clock_t start = clock();
	printf("--------------------------------\n");
	// for (a = 0; a < num_threads; a++) pt[a] = boost::thread(TrainLINEThread, (void *)a);
	for (a = 0; a < num_threads; a++) pt[a] = std::thread(TrainLINEThread, a);
	for (a = 0; a < num_threads; a++) pt[a].join();
	printf("\n");
	clock_t finish = clock();
	printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);

	free(neg_table_v);
	free(neg_table_r);
	free(neg_table_t);
	free(neg_table_w);
	Output();
	std::cout<<"GE finish..... "<<"\n";
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	if (argc == 1) {
		printf("LINE: Large Information Network Embedding\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the learnt embeddings in binary moded; default is 0 (off)\n");
		printf("\t-size <int>\n");
		printf("\t\tSet dimension of vertex embeddings; default is 100\n");
		printf("\t-order <int>\n");
		printf("\t\tNumber of negative examples; default is 5\n");
		printf("\t-samples <int>\n");
		printf("\t\tSet the number of training samples as <int>Million; default is 1\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 1)\n");
		printf("\t-rho <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025\n");
		printf("\nExamples:\n");
		printf("./GEmodel -binary 1 -size 200 -order 2 -negative 5 -samples 100 -rho 0.025 -threads 20\n\n");
		return 0;
	}

	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) is_binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	
	strcpy(poi_file, "POIs.txt");

	strcpy(net_poi, "net_POI.txt"); 
	strcpy(emb_poi, "net_POI_vec.txt");
	strcpy(net_poi_reg, "net_POI_reg.txt"); 
	strcpy(emb_reg, "net_reg_vec.txt");
	strcpy(net_poi_time, "net_POI_time.txt"); 
	strcpy(emb_time, "net_time_vec.txt");
	strcpy(net_poi_word, "net_POI_word.txt"); 
	strcpy(emb_word, "net_word_vec.txt");
	total_samples *= 1000000;
	rho = init_rho;
	vertex_poi = (struct ClassVertex *)calloc(max_num_vertices, sizeof(struct ClassVertex));
	vertex_v = (struct ClassVertex *)calloc(max_num_vertices, sizeof(struct ClassVertex));
	vertex_r = (struct ClassVertex *)calloc(max_num_vertices, sizeof(struct ClassVertex));
	vertex_t = (struct ClassVertex *)calloc(max_num_vertices, sizeof(struct ClassVertex));
	vertex_w = (struct ClassVertex *)calloc(max_num_vertices, sizeof(struct ClassVertex));
	TrainLINE();
	return 0;
}