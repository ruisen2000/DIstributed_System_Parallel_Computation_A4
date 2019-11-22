#include <iostream>
#include "core/utils.h"
#include "core/graph.h"
#include <mpi.h>

uintV countTriangles(uintV *array1, uintE len1, uintV *array2, uintE len2, uintV u, uintV v)
{

    uintE i = 0, j = 0; // indexes for array1 and array2
    uintV count = 0;

    if (u == v)
        return count;

    while ((i < len1) && (j < len2))
    {
        if (array1[i] == array2[j])
        {
            if ((array1[i] != u) && (array1[i] != v))
            {
                count++;
            }
            else
            {
                // triangle with self-referential edge -> ignore
            }
            i++;
            j++;
        }
        else if (array1[i] < array2[j])
        {
            i++;
        }
        else
        {
            j++;
        }
    }
    return count;
}

void triangleCountSerial(Graph &g)
{
    uintV n = g.n_;
    long triangle_count = 0;
    double time_taken;
    timer t1;
    t1.start();
    for (uintV u = 0; u < n; u++)
    {
        uintE out_degree = g.vertices_[u].getOutDegree();
        for (uintE i = 0; i < out_degree; i++)
        {
            uintV v = g.vertices_[u].getOutNeighbor(i);
            triangle_count += countTriangles(g.vertices_[u].getInNeighbors(),
                                             g.vertices_[u].getInDegree(),
                                             g.vertices_[v].getOutNeighbors(),
                                             g.vertices_[v].getOutDegree(),
                                             u,
                                             v);
        }
    }

    // For every thread, print out the following statistics:
    // rank, edges, triangle_count, communication_time
    // 0, 17248443, 144441858, 0.000074
    // 1, 17248443, 152103585, 0.000020
    // 2, 17248443, 225182666, 0.000034
    // 3, 17248444, 185596640, 0.000022

    time_taken = t1.stop();

    // Print out overall statistics
    std::cout << "Number of triangles : " << triangle_count << "\n";
    std::cout << "Number of unique triangles : " << triangle_count / 3 << "\n";
    std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION) << time_taken << "\n";
}

void getStartEndValues(Graph &g, uintV n, uint* assignedVerticies, int n_workers)
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

void triangleCountParallel(Graph &g, uint strategy)
{
    uintV n = g.n_;

    int process_id;
    int num_processors;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    uint* assignedVerticies = new uint[num_processors];  // Array stores the start/end vertex for each thread
    getStartEndValues(g, n, assignedVerticies, num_processors - 1);

    uintV start = assignedVerticies[process_id - 1];
    uintV end = assignedVerticies[process_id];
    long count = 0; 
    long global_count = 0;
    uintE edge_count = 0;
    double sync_time = 0.0;

    timer t1;

    if(process_id != 0){
        for (uintV u = start; u < end; u++)
        {
            uintE out_degree = g.vertices_[u].getOutDegree();
            edge_count += out_degree;
            for (uintE i = 0; i < out_degree; i++)
            {
                uintV v = g.vertices_[u].getOutNeighbor(i);
                count += countTriangles(g.vertices_[u].getInNeighbors(),
                                                 g.vertices_[u].getInDegree(),
                                                 g.vertices_[v].getOutNeighbors(),
                                                 g.vertices_[v].getOutDegree(),
                                                 u,
                                                 v);            
            }
        }

        t1.start();

        if (strategy == 1)
        {
            MPI_Send(&count, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
        }
        else if (strategy == 2)
        {
            MPI_Gather(&count, 1, MPI_LONG, NULL, 0, MPI_LONG, 0, MPI_COMM_WORLD);
        }
        else if (strategy == 3)
        {
            MPI_Reduce(&count, NULL, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        sync_time = t1.stop();
    }

    // --- synchronization phase start ---
    if (process_id == 0){

        std::cout << "Communication strategy : " << strategy << "\n";
        std::cout << "World size : " << num_processors << "\n";
        std::cout << "rank, edges, triangle_count, communication_time" << std::endl;

        t1.start();
        if (strategy == 1)
        {
            for (int i = 1; i < num_processors; i++)
            {
                MPI_Recv(&count, 1, MPI_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                global_count = global_count + count;
            }
        }
        else if (strategy == 2)
        {
            long* buffer = (long *)malloc(sizeof(long) * num_processors);
            MPI_Gather(&count, 1, MPI_LONG, buffer, 1, MPI_LONG, 0, MPI_COMM_WORLD);
            for (int i = 1; i < num_processors; i++)
            {
                global_count = global_count + buffer[i];
            }
            free(buffer);
        }
        else if (strategy == 3)
        {
            MPI_Reduce(&count, &global_count, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        sync_time = t1.stop();
    }

    // --- synchronization phase end -----

    if(process_id == 0){
        // Print out overall statistics
        std::cout << "Number of triangles : " << global_count << "\n";
        std::cout << "Number of unique triangles : " << global_count / 3 << "\n";
        
        // print process statistics and other results
    }
    else{
        // print process statistics
        std::cout << process_id << ", " << edge_count << ", " count << ", " << sync_time;
    }
}


int main(int argc, char *argv[])
{
    cxxopts::Options options("triangle_counting_serial", "Count the number of triangles using serial and parallel execution");
    options.add_options("custom", {
                                      {"strategy", "Strategy to be used", cxxopts::value<uint>()->default_value(DEFAULT_STRATEGY)},
                                      {"inputFile", "Input graph file path", cxxopts::value<std::string>()->default_value("/scratch/assignment1/input_graphs/roadNet-CA")},
                                  });

    auto cl_options = options.parse(argc, argv);
    uint strategy = cl_options["strategy"].as<uint>();
    std::string input_file_path = cl_options["inputFile"].as<std::string>();

    MPI_Init(NULL, NULL);

    std::cout << std::fixed;
    // Get the world size and print it out here
    // 
    

    Graph g;
    g.readGraphFromBinary<int>(input_file_path);

    timer t1;
    t1.start();
    switch (strategy)
    {
    case 0:
        triangleCountSerial(g);
        break;
    case 1:
        triangleCountParallel(g, 1);
        break;
    case 2:
        triangleCountParallel(g, 2);
        break;
    case 3:
        triangleCountParallel(g, 3);
        break;
    default:
        break;
    }

    int process_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    if (process_id == 0)
    {
        double time_taken = t1.stop();
        std::cout << "Time taken (in seconds) : " << std::setprecision(TIME_PRECISION) << time_taken << "\n";
    }    

    MPI_Finalize();

    return 0;
}

