#include <iostream>
#include <mpi.h>
#include "core/utils.h"
#include "core/graph.h"

#ifdef USE_INT
#define INIT_PAGE_RANK 100000
#define EPSILON 1000
#define PAGE_RANK(x) (15000 + (5 * x) / 6)
#define CHANGE_IN_PAGE_RANK(x, y) std::abs(x - y)
#define PAGERANK_MPI_TYPE MPI_LONG
typedef int64_t PageRankType;
#else
#define INIT_PAGE_RANK 1.0
#define EPSILON 0.01
#define DAMPING 0.85
#define PAGE_RANK(x) (1 - DAMPING + DAMPING * x)
#define CHANGE_IN_PAGE_RANK(x, y) std::fabs(x - y)
#define PAGERANK_MPI_TYPE MPI_FLOAT
typedef float PageRankType;
#endif

/* Serial Computation using 1 Process */
void pageRankSerial(Graph &g, int max_iters)
{
    uintV n = g.n_;
    double time_taken;
    timer t1;
    PageRankType *pr_curr = new PageRankType[n];
    PageRankType *pr_next = new PageRankType[n];

    int process_id;
        MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    t1.start();
    for (uintV i = 0; i < n; i++)
    {
        pr_curr[i] = INIT_PAGE_RANK;
        pr_next[i] = 0.0;
    }

    // Push based pagerank
    // -------------------------------------------------------------------
    for (int iter = 0; iter < max_iters; iter++)
    {
        // for each vertex 'u', process all its outNeighbors 'v'
        for (uintV u = 0; u < n; u++)
        {
            uintE out_degree = g.vertices_[u].getOutDegree();
            for (uintE i = 0; i < out_degree; i++)
            {
                uintV v = g.vertices_[u].getOutNeighbor(i);
                pr_next[v] += (pr_curr[u] / out_degree);
            }
        }
                
        for (uintV v = 0; v < n; v++)
        {
            pr_next[v] = PAGE_RANK(pr_next[v]);

            // reset pr_curr for the next iteration
            pr_curr[v] = pr_next[v];
            pr_next[v] = 0.0;
        }
        
    }
    // -------------------------------------------------------------------

    // For every thread, print the following statistics:
    // rank, num_edges, communication_time
    // 0, 344968860, 1.297778
    // 1, 344968860, 1.247763
    // 2, 344968860, 0.956243
    // 3, 344968880, 0.467028

    PageRankType sum_of_page_ranks = 0;
    for (uintV u = 0; u < n; u++)
    {
        sum_of_page_ranks += pr_curr[u];
    }
    time_taken = t1.stop();
    std::cout << "Sum of page rank : " << sum_of_page_ranks << "\n";
    std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION) << time_taken << "\n";
    delete[] pr_curr;
    delete[] pr_next;
}

/* Assign parts of the array to each MPI call */
void getStartEndValues(Graph &g, uintV n, int* assignedVerticies, int n_workers)
{
    int arrLen = n_workers + 1;

    for (int i = 0; i < arrLen; i++)
    {
        assignedVerticies[i] = 0;
    }

    assignedVerticies[arrLen - 1] = n;


    // Count the total number of out degrees (edges), then find verticies such that
    // each get 1/4 of all the edges
    uintE total_edges = 0;
    for (uintV u = 0; u < n; u++)
    {
        total_edges += g.vertices_[u].getOutDegree();
    }

    // find verticies such that each have 1/4 of all the edges between them
    uint cutOffValue = total_edges / n_workers;
    int counter = 1;  // count which element of assignedVerticies array we're calculating
    total_edges = 0;
    for (uintV u = 0; u < n; u++)
    {
        total_edges += g.vertices_[u].getOutDegree();
        if (total_edges > cutOffValue * counter)  // total > 1/4 edges, 1/2 edges, 3/4 edges
        {           
            assignedVerticies[counter] = u;
            counter++;

            if (counter == n_workers)
            {
                break;
            }
        }
    }
}

/* Use MPI System: 
Strategy 1: Use MPI_SEND/MPI_RECEIVE to send to each individual process for computation.
Strategy 2: Use MPI_Gather to gather results back to process 0.
Strategy 3: Use MPI_Reduce to automatically combine results.
*/
void pageRankParallel(Graph &g, int max_iters, uint strategy)
{

    uintV n = g.n_;

    int process_id;
    int num_processors;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    PageRankType *pr_curr = new PageRankType[n];
    PageRankType *pr_next = new PageRankType[n];
    PageRankType *buffer = new PageRankType[n];

    for (uintV i = 0; i < n; i++)
    {
        pr_curr[i] = INIT_PAGE_RANK;
        pr_next[i] = 0.0;
        buffer[i] = 0.0;
    }

    int* assignedVerticies = new int[num_processors + 1];  // Array stores the start vertex for each thread (where start of next process = end of prev process)
    getStartEndValues(g, n, assignedVerticies, num_processors);

    int start = assignedVerticies[process_id];
    int end = assignedVerticies[process_id + 1];
    

    int* sendCount = new int[num_processors];
    sendCount[0] = 0;
    for (int i = 1; i < num_processors; i++)
    {
       sendCount[i] = assignedVerticies[i + 1] - assignedVerticies[i];
    }

    uintE edges_processed = 0;
    PageRankType expected = 0;

    timer t1;
    timer t2;
    double total_time = 0.0;
    double sync_time = 0.0;

    t2.start();    

    /* Process assigned part of the graph */
    for (int iter = 0; iter < max_iters; iter++)
    {        
        // for each vertex 'u', process all its outNeighbors 'v'
        for (uintV u = start; u < end; u++)
        {
            uintE out_degree = g.vertices_[u].getOutDegree();
            edges_processed += out_degree;
            for (uintE i = 0; i < out_degree; i++)
            {
                uintV v = g.vertices_[u].getOutNeighbor(i);                
                pr_next[v] += (pr_curr[u] / out_degree);                
            }
        }    

        t1.start();
        if(process_id != 0) {     // Worker Process       
            if (strategy == 1)
            {
                int numItems = end - start;
                MPI_Send(pr_next, n, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer, numItems, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for (int i = start; i < end; i++)
                {
                    pr_next[i] = buffer[i - start];
                }
                //std::cout << std::endl;
            }
             else if (strategy == 2)
            {
               MPI_Reduce(pr_next, NULL, n, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
               MPI_Scatterv(NULL, sendCount, assignedVerticies, MPI_LONG_LONG, buffer, end - start, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
            
               for (int i = start; i < end; i++)
                {
                    pr_next[i] = buffer[i - start];
                }
            }
            else if (strategy == 3)
            {
                for (int i = 0; i < num_processors; i++)
                {
                    uintV p_start = assignedVerticies[i];
                    uintV p_end = assignedVerticies[i + 1];
                    if (i == process_id)
                    {
                        // receive chunks from other processes
                        MPI_Reduce(pr_next + start, buffer, end - start, MPI_LONG_LONG, MPI_SUM, process_id, MPI_COMM_WORLD);
                        for (int i = start; i < end; i++)
                        {
                            pr_next[i] = buffer[i - start];
                        }
                    }
                    else
                    {
                        // send chunks to other processes
                        MPI_Reduce(pr_next + p_start, NULL, p_end - p_start, MPI_LONG_LONG, MPI_SUM, i, MPI_COMM_WORLD);    
                    }
                    
                }
            }
            else
            {
                std::cout << "invalid strategy: " << strategy;
            }

        } 
        else // Root process, gather computed results from workers        
        {            
            if (strategy == 1)
            {
                for (int i = 1; i < num_processors; i++)
                {
                    MPI_Recv(buffer, n, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (int j = 0; j < n; j++)
                    {
                        pr_next[j] = pr_next[j] + buffer[j];
                    }
                }                              

                for (int i = 1; i < num_processors; i++)
                {
                    uintV p_start = assignedVerticies[i];
                    uintV p_end = assignedVerticies[i + 1];
                    //std::cout << "send to process " << i << " start: " << p_start << " end " << p_end << " n: " << n << std::endl;
                    MPI_Send(pr_next + p_start, p_end - p_start, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD);
                    
                }
            }
            else if (strategy == 2)
            {               
               MPI_Reduce(pr_next, buffer, n, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
               MPI_Scatterv(buffer, sendCount, assignedVerticies, MPI_LONG_LONG, NULL, 0, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

               for (int i = start; i < end; i++)
               {
                    pr_next[i] = buffer[i - start];
               }
            }
            else if (strategy == 3)
            {                
                for (int i = 0; i < num_processors; i++)
                {
                    uintV p_start = assignedVerticies[i];
                    uintV p_end = assignedVerticies[i + 1];
                    MPI_Reduce(pr_next + p_start, buffer, p_end - p_start, MPI_LONG_LONG, MPI_SUM, i, MPI_COMM_WORLD);

                    if (i == 0)
                    {
                        for (int i = start; i < end; i++)
                        {
                            pr_next[i] = buffer[i - start];
                        }
                    }
                }
            }
            else
            {
                std::cout << "invalid strategy: " << strategy;
            }
        }
        sync_time = t1.stop();

        for (uintV v = start; v < end; v++)
        {       
            pr_next[v] = PAGE_RANK(pr_next[v]);

            // reset pr_curr for the next iteration
            pr_curr[v] = pr_next[v];                      
        }
        for (int i = 0; i < n; i++)
        {
            pr_next[i] = 0;
        }
    }
    
    // Individual Process stats
    std::cout << process_id << ", " << edges_processed << ", " << sync_time << std::endl;

    PageRankType sum_of_page_ranks = 0;

    for (uintV u = start; u < end; u++)
    {
        sum_of_page_ranks += pr_curr[u];
    }

    if (process_id != 0)
    {        
        // send sum to root
        MPI_Reduce(&sum_of_page_ranks, NULL, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);        
    }
    else
    {  // root process, cumultative stats
        PageRankType global_sum = 0;
        MPI_Reduce(&sum_of_page_ranks, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        std::cout << "Sum of page rank : " << global_sum << "\n";               
    }

    total_time = t2.stop();
    delete[] pr_curr;
    delete[] pr_next;
    delete[] buffer;
    delete[] assignedVerticies;
    delete[] sendCount;
 
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("page_rank_push", "Calculate page_rank using serial and parallel execution");
    options.add_options("", {
                                {"nIterations", "Maximum number of iterations", cxxopts::value<uint>()->default_value(DEFAULT_MAX_ITER)},
                                {"strategy", "Strategy to be used", cxxopts::value<uint>()->default_value(DEFAULT_STRATEGY)},
                                {"inputFile", "Input graph file path", cxxopts::value<std::string>()->default_value("/scratch/assignment1/input_graphs/roadNet-CA")},
                            });

    auto cl_options = options.parse(argc, argv);
    uint strategy = cl_options["strategy"].as<uint>();
    uint max_iterations = cl_options["nIterations"].as<uint>();
    std::string input_file_path = cl_options["inputFile"].as<std::string>();

    MPI_Init(NULL, NULL);

    int process_id;
    int num_processors;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    if(process_id == 0)
    {
    #ifdef USE_INT
        std::cout << "Using INT\n";
    #else
        std::cout << "Using FLOAT\n";
    #endif
        std::cout << std::fixed;
        // Get the world size and print it out here
        std::cout << "World size : " << num_processors << "\n";
        std::cout << "Communication strategy : " << strategy << "\n";
        std::cout << "Iterations : " << max_iterations << "\n";
        std::cout << "rank, num_edges, communication_time" << std::endl;
    }

    Graph g;
    g.readGraphFromBinary<int>(input_file_path);

    timer t1;
    t1.start();

    switch (strategy)
    {
    case 0:
        pageRankSerial(g, max_iterations);
        break;
    case 1:
        pageRankParallel(g, max_iterations, strategy);
        break;
    case 2:
        pageRankParallel(g, max_iterations, strategy);
        break;
    case 3:
        pageRankParallel(g, max_iterations, strategy);
        break;
    default:
        break;
    }

    if (process_id == 0)
    {
        double time_taken = t1.stop();
        std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION) << time_taken << "\n";
    }    


    MPI_Finalize();
    return 0;
}
