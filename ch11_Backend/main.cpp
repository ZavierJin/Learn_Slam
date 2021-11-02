/*
 * Optimize the pose diagram with g2o solver
 * Sphere.g2o is a manually generated pose graph
 * Although the entire graph can be read directly through the load function,
 * we still implement the reading code ourselves in order to gain a deeper understanding
 * Se3 in g2o / types / slam3d / is used to represent the pose,
 * which is essentially a quaternion rather than a lie algebra
 */

#include <iostream>
#include <fstream>
#include <string>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#define DATA_PATH "sphere.g2o"

int main()
{
    std::ifstream fin(DATA_PATH);
    if (!fin) {
        std::cout << "file " << DATA_PATH << " does not exist." << std::endl;
        return 1;
    }

    // g2o setting
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;         // graph model
    optimizer.setAlgorithm(solver);         // set solver
    optimizer.setVerbose(true);      // open optimizing detail

    int vertexCnt = 0, edgeCnt = 0;         // vertex and edge count
    while (!fin.eof()) {
        std::string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
            // SE3 vertex
            auto* v = new g2o::VertexSE3();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt++;
            if (index == 0)
                v->setFixed(true);
        } else if (name == "EDGE_SE3:QUAT") {
            // SE3-SE3 edge
            auto* e = new g2o::EdgeSE3();
            int idx1, idx2;     // associate two vertices
            fin >> idx1 >> idx2;
            e->setId(edgeCnt++);
            e->setVertex(0, optimizer.vertices()[idx1]);
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin);
            optimizer.addEdge(e);
        }
        if (!fin.good()) 
            break;
    }

    std::cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << std::endl;
    std::cout << "optimizing ..." << std::endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);
    std::cout << "saving optimization results ..." << std::endl;
    optimizer.save("result.g2o");
    return 0;
}