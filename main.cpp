#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include <optional>
#include <memory>
#include <unordered_map>
#include <map>
#include <queue>
#include <limits>
#include <set>

// Work of Sinii Viacheslav

/** test cases:
  * add already added edge/vertex,
  * delete non-existing edge/vertex
  * output empty graph
  * find deleted edge/vertex
  * find edge from a vertex after deleting this vertex
  * make cycle between two adjacent vertices
  * find shortest path with bandwidth lower than the bandwidth of all edges in a graph
  * add edge between non-existing vertices
*/
using namespace std;

namespace exceptions
{
    class ActionWithNonExistingVertexException : public exception
    {
        [[nodiscard]] const char* what() const noexcept override
        {
            return "ActionWithNonExistingVertexException!";
        }
    };

    class DuplicateEdgeException : public exception
    {
        [[nodiscard]] const char* what() const noexcept override
        {
            return "DuplicateEdgeException!";
        }
    };
}

// class with info and method for vertex
template <typename T>
class Vertex
{
public:
    Vertex(unique_ptr<T> in_value, int m_ind)
        :
        value(move(in_value)),
        index_in_matrix(m_ind)
    {}

    void printInfo() const
    {
        cout << "Vertex: " << index_in_matrix
             << " value: " << *value << endl;
    }

public:
    [[nodiscard]] T& getValue() const {
        return *value;
    }

    void setValue(T new_value) {
        Vertex::value = new_value;
    }

    [[nodiscard]] int getIndexInMatrix() const {
        return index_in_matrix;
    }

    void setIndexInMatrix(int indexInMatrix) {
        index_in_matrix = indexInMatrix;
    }

private:
    unique_ptr<T> value;
    int index_in_matrix = -1;
};

// class with info and method for edge
template <typename N, typename T>
class Edge
{
public:
    Edge(shared_ptr<Vertex<T>>& out, shared_ptr<Vertex<T>>& in, N& weight)
        :
        origin(out),
        destination(in),
        weight(weight)
    {}

    Edge(shared_ptr<Vertex<T>>& out, shared_ptr<Vertex<T>>& in, N& weight, N& bandwidth)
        :
        origin(out),
        destination(in),
        weight(weight),
        bandwidth(bandwidth)
    {}

    void printInfo() const
    {
        cout << "Edge: " << origin->getValue()
             << "-" << destination->getValue()
             << " weight: " << weight << endl;
    }
public:
    [[nodiscard]] N getWeight() const {
        return weight;
    }

    void setWeight(N new_weight) {
        weight = new_weight;
    }

    N getBandwidth() const {
        return bandwidth;
    }

    void setBandwidth(N bandwidth) {
        Edge::bandwidth = bandwidth;
    }

    [[nodiscard]] const shared_ptr<Vertex<T>> &getOrigin() const {
        return origin;
    }

    void setOrigin(const shared_ptr<Vertex<T>> &new_origin) {
        Edge::origin = new_origin;
    }

    [[nodiscard]] const shared_ptr<Vertex<T>> &getDestination() const {
        return destination;
    }

    void setDestination(const shared_ptr<Vertex<T>> &new_destination) {
        Edge::destination = new_destination;
    }

private:
    N weight;
    N bandwidth;
    shared_ptr<Vertex<T>> origin;
    shared_ptr<Vertex<T>> destination;
};

// graph interface
template <typename T, typename N>
class GraphADT
{
public:
    virtual shared_ptr<Vertex<T>> addVertex(const T&) = 0;
    virtual void removeVertex(shared_ptr<Vertex<T>>) = 0;
    virtual shared_ptr<Edge<N, T>> addEdge(shared_ptr<Vertex<T>>, shared_ptr<Vertex<T>>, N) = 0;
    virtual void removeEdge(shared_ptr<Edge<N, T>>) = 0;
    [[nodiscard]] virtual optional<list<shared_ptr<Edge<N, T>>>> edgesFrom(const shared_ptr<Vertex<T>>& vertex) const = 0;
    [[nodiscard]] virtual optional<list<shared_ptr<Edge<N, T>>>> edgesTo(const shared_ptr<Vertex<T>>& vertex) const = 0;
    [[nodiscard]] virtual optional<shared_ptr<Vertex<T>>> findVertex(const T&) const = 0;
    [[nodiscard]] virtual optional<shared_ptr<Edge<N, T>>> findEdge(const T& from, const T& to) const = 0;
    [[nodiscard]] virtual bool hasEdge(const shared_ptr<Vertex<T>>&, const shared_ptr<Vertex<T>>&) const = 0;
};

// my graph class
template <typename T, typename N>
class AdjacencyMatrixGraph : GraphADT<T, N> {
public:
    // graph constructor
    AdjacencyMatrixGraph()
            :
            greatest_occupied_index(-1),
            matrix(initial_matrix_size, vector<shared_ptr<Edge<N, T>>>(initial_matrix_size, nullptr))
    {
        for (int i = 0; i < initial_matrix_size; ++i)
        {
            free_vertices.push(i);
        }
    }

    /* Add new vertex */
private:
    // assigns the lowest available vertex to the new vertex if matrix is not full
    inline int simpleAdd()
    {
        // extract the lowest available index
        int index = free_vertices.top();
        free_vertices.pop();

        // check if it is new greatest occupied vertex
        if (index > greatest_occupied_index)
            greatest_occupied_index = index;

        // return assigned index
        return index;
    }

    // if the graph is full - we double its size and then return assigned index
    inline int doubleMatrix()
    {
        // assign the value of index
        int index = matrix.size();
        // as it guaranteed will be the highest - update greatest_occupied_index
        greatest_occupied_index = index;

        // double the size of the matrix
        int newSize = matrix.size() * 2;

        // push all newly available vertices to the set of available vertices
        for (int i = greatest_occupied_index + 1; i < newSize; ++i) {
            free_vertices.push(i);
        }

        // double the number of rows
        matrix.resize(newSize);
        // double the number of columns
        for (int i = 0; i < newSize; ++i) {
            matrix[i].resize(newSize);
        }

        // return assigned index
        return index;
    }

    // return assigned index
    inline int newElementIndex()
    {
        // simpleAdd() - if we have free indices.
        // Otherwise, call doubleMatrix()
        return !free_vertices.empty() ? simpleAdd() : doubleMatrix();
    }

public:
    // adds new vertex to the graph
    shared_ptr<Vertex<T>> addVertex(const T& value) override
    {
        // what to do duplicate names of vertices
        if (findVertex(value)) return *findVertex(value);

        // obtain index for new vertex
        int index = newElementIndex();

        // create new vertex
        auto new_vertex = make_shared<Vertex<T>>(Vertex<T>(make_unique<T>(value), index));
        // insert new vertex to the list of vertices
        auto it = vertexList.insert({ value, new_vertex });

//        printInfo();

        // return reference to new vertex
        return new_vertex;
    }

    /* End add new vertex */

    /* Remove vertex */
private:
    // delete all incident edges
    inline void deleteIncidentEdges(shared_ptr<Vertex<T>> vertex)
    {
        // delete outgoing edges
        if (auto edges = edgesFrom(vertex))
            for (const auto& tmp : *edges)
                removeEdge(tmp);

        // delete incoming edges
        if (auto edges = edgesTo(vertex))
            for (const auto& tmp : *edges)
                removeEdge(tmp);
    }

    // clear according row and column
    inline void cleanUp(int index)
    {
        for (int i = 0; i < greatest_occupied_index + 1; ++i) {
            matrix[index][i] = nullptr;
            matrix[i][index] = nullptr;
        }
    }

public:
    // function that removes a vertex from the graph
    void removeVertex(shared_ptr<Vertex<T>> vertex) override
    {
        // obtain the index of this vertex in the matrix
        int index = vertex->getIndexInMatrix();

        if (vertexList.find(vertex->getValue()) == vertexList.end()) return;
//            throw exceptions::ActionWithNonExistingVertexException();

        // add freed index to the set of free indices
        free_vertices.push(index);

        // if it was the greatest_occupied_index - decrement the latter
        if (index == greatest_occupied_index)
            --greatest_occupied_index;

        // clear according row and column
        cleanUp(index);

        // delete vertex from the vertex list
        vertexList.erase(vertex->getValue());

        // delete all edges incident to the vertex
        deleteIncidentEdges(vertex);

//        printInfo();
    }

    /* end remove vertex */

    /* Add new edge */

private:
    // check if the client tries to add edge with non-existing end vertices
    inline void nonExistingEndVerticesCheck(T fromValue, T toValue)
    {
        if (vertexList.find(fromValue) == vertexList.end()
            || vertexList.find(toValue) == vertexList.end())
            throw exceptions::ActionWithNonExistingVertexException();
    }

    // manage adding new edge to the matrix and edge list.
    // check on adding new edge twice and adding edge with non-existing end vertices
    void manageEdgeAddition(shared_ptr<Edge<N, T>> new_edge)
    {
        // obtain end vertices
        auto fromVertex = new_edge->getOrigin();
        auto toVertex = new_edge->getDestination();

        // obtain values of end vertices
        auto fromValue = fromVertex->getValue();
        auto toValue = toVertex->getValue();

        nonExistingEndVerticesCheck(fromValue, toValue);

        // TODO: adding already existing edge
        // if the client tries to add already existing edge - just update its weight
        if (auto edge = findEdge(fromValue, toValue))
            (*edge)->setWeight(new_edge->getWeight());

        // add the edge to the list of edges
        auto it = edgeList.insert({{ fromVertex, toVertex }, new_edge });

        // fill the according place in the matrix
        matrix[fromVertex->getIndexInMatrix()][toVertex->getIndexInMatrix()] = new_edge;
    }

public:
    // add new edge with bandwidth to the graph
    shared_ptr<Edge<N, T>> addEdge(shared_ptr<Vertex<T>> from, shared_ptr<Vertex<T>> to, N weight) override
    {
        // construct new edge
        auto new_edge = make_shared<Edge<N, T>>(
                Edge<N, T>(
                        from, to, weight
                )
        );

        manageEdgeAddition(new_edge);

//        printInfo();

        // return a reference to the newly created edge object
        return new_edge;
    }

    // add new edge without bandwidth to the graph
    shared_ptr<Edge<N, T>> addEdge(shared_ptr<Vertex<T>> from, shared_ptr<Vertex<T>> to, N weight, N bandwidth)
    {
        auto new_edge = make_shared<Edge<N, T>>(
                Edge<N, T>(
                        from, to, weight, bandwidth
                )
        );

        manageEdgeAddition(new_edge);

//        printInfo();
        return new_edge;
    }

    /* End add new edge */

    /* Remove an edge */

    // remove an edge from the graph
    void removeEdge(shared_ptr<Edge<N, T>> edge) override
    {
        // obtain end vertices
        auto fromVertex = edge->getOrigin();
        auto toVertex = edge->getDestination();

        // delete the edge from the edge list
        edgeList.erase({ fromVertex, toVertex });

        // empty the place of this edge in the matrix
        matrix[fromVertex->getIndexInMatrix()][toVertex->getIndexInMatrix()] = nullptr;

//        printInfo();
    }

    /* End remove an edge */

    // obtain the list of edges outgoing from the specified vertex
    optional<list<shared_ptr<Edge<N, T>>>> edgesFrom(const shared_ptr<Vertex<T>>& vertex) const override
    {
        // if a client tries to find edges from non-existing vertex
        if (vertexList.find(vertex->getValue()) == vertexList.end()) return {};
//            throw exceptions::ActionWithNonExistingVertexException();

        list<shared_ptr<Edge<N, T>>> edges;
        // obtain vertex's index in the matrix
        int index = vertex->getIndexInMatrix();

        // go through each cell in the vertex’s row in the matrix
        // and collect all non-nullptr values into a list
        for (int i = 0; i < greatest_occupied_index + 1; ++i)
        {
            if (matrix[index][i] != nullptr)
            {
                edges.push_back(matrix[index][i]);
            }
        }

        // if there is no outgoing edges - return nullopt
        if (edges.size() == 0) return {};
        else return edges; // otherwise, return the list
    }

    // obtain the list of edges incoming to the specified vertex
    optional<list<shared_ptr<Edge<N, T>>>> edgesTo(const shared_ptr<Vertex<T>>& vertex) const override
    {
        if (vertexList.find(vertex->getValue()) == vertexList.end()) return {};
//            throw exceptions::ActionWithNonExistingVertexException();

        list<shared_ptr<Edge<N, T>>> edges;
        // obtain vertex's index in the matrix
        int index = vertex->getIndexInMatrix();

        // go through each cell in the vertex’s column in the matrix
        // and collect all non-nullptr values into a list
        for (int i = 0; i < greatest_occupied_index + 1; ++i)
        {
            if (matrix[i][index] != nullptr)
            {
                edges.push_back(matrix[i][index]);
            }
        }

        // if there is no incoming edges - return nullopt
        if (edges.size() == 0) return {};
        else return edges; // otherwise, return the list
    }

    // find a vertex with the specified value
    optional<shared_ptr<Vertex<T>>> findVertex(const T& value) const override
    {
        // try to find a vertex with the specified value in the vertex list
        auto tmp = vertexList.find(value);

        // if the vertex is found - return a reference to it
        if (tmp != vertexList.end())
            return tmp->second;
        else return {}; // if there is no such vertex - return nullopt
    }

    // find an edge with the specified values at end vertices
    optional<shared_ptr<Edge<N, T>>> findEdge(const T& from, const T& to) const override
    {
        // try to obtain vertices with the specified values in the graph
        const auto v1 = findVertex(from);
        const auto v2 = findVertex(to);

        // if the vertices were found
        if (v1 && v2)
        {
            // try to find an edge with the same end vertices in the edge list
            const auto tmp = edgeList.find({ v1.value(), v2.value() });

            // if the edge is found - return a reference to it
            if (tmp != edgeList.end())
                return tmp->second;
        }

        return {}; // if there is no such edge - return nullopt
    }

    // check if there is an edge between two vertices
    bool hasEdge(const shared_ptr<Vertex<T>>& from, const shared_ptr<Vertex<T>>& to) const override
    {
        // check if there are such vertices in the graph
        if (vertexList.find(from->getValue()) == vertexList.end() || vertexList.find(to->getValue()) == vertexList.end())
            throw exceptions::ActionWithNonExistingVertexException();

        // true if this function could find the edge in the graph
        return findEdge(from->getValue(), to->getValue()).has_value();
    }

    /* isAcyclic */
private:
    // indicate whether a vertex was visited during dfs
    enum class Colors
    {
        White,
        Grey,
        Black
    };

    // info about vertex needed for successful dfs
    struct VertexInDFS
    {
        shared_ptr<Vertex<T>> vertex;
        Colors color;
        shared_ptr<Vertex<T>> parent;
    };

    // initialize vertices for using dfs
    void initVerticesForDFS(vector<VertexInDFS>& verticesInfo) const
    {
        for (const auto& tmp : vertexList)
        {
            // initial value for each vertex -
            // vertex itself, white color, and no parent
            verticesInfo[tmp.second->getIndexInMatrix()] = { tmp.second, Colors::White, nullptr };
        }
    }

    // reproduce a cyclic path
    vector<shared_ptr<Vertex<T>>>* constructPath(pair<int, int> cycleStartEnd, vector<VertexInDFS>& verticesInfo) const
    {
        // starting vertex in a path
        int current = cycleStartEnd.first;

        // the path is contained in a vector
        auto path = new vector<shared_ptr<Vertex<T>>>();
        path->push_back(verticesInfo[current].vertex);

        // go through each vertex in a path
        while (current != cycleStartEnd.second)
        {
            // put a vertex into the path vector
            auto parent = verticesInfo[current].parent;
            path->push_back(parent);
            // go to next vertex
            current = parent->getIndexInMatrix();
        }

        // make it in right order
        reverse(path->begin(), path->end());

        return path;
    }

    // find total weight of a path
    N calculateTotalWeight(const vector<shared_ptr<Vertex<T>>>& path) const
    {
        // extract an edge between two ends in a path
        auto pathEndsEdge = *findEdge(path[path.size() - 1]->getValue(), path[0]->getValue());
        // initialize variable that stores total weight
        N totalWeight = pathEndsEdge->getWeight();

        // go through each edge in a path and calculate total weight
        for (int i = 0; i < path.size() - 1; ++i)
        {
            auto edge = *findEdge(path[i]->getValue(), path[i + 1]->getValue());
            totalWeight += edge->getWeight();
        }

        return totalWeight;
    }

    // depth-first search in a graph
    optional<pair<int, int>> dfs_visit(const shared_ptr<Vertex<T>>& u, vector<VertexInDFS>& vertexInfo) const
    {
        // mark it visited
        vertexInfo[u->getIndexInMatrix()].color = Colors::Grey;

        // initialize result
        optional<pair<int, int>> result = nullopt;

        // if a vertex contains outgoing edges
        if (auto edges = edgesFrom(u))
            // go through each edge outgoing from a vertex
            for (const auto& edge : *edges)
            {
                // extract neighbour vertex
                auto v = edge->getDestination();

                // if a vertex is already visited
                if (vertexInfo[v->getIndexInMatrix()].color == Colors::Grey)
                {
                    // return start and end vertices in the cyclic path
                    return make_pair(u->getIndexInMatrix(), v->getIndexInMatrix());
                }
                else if (vertexInfo[v->getIndexInMatrix()].color == Colors::White) // if not yet visited
                {
                    // mark current vertex a parent for this neighbour
                    vertexInfo[v->getIndexInMatrix()].parent = u;

                    // if we found a cycle - return
                    if (!(result = dfs_visit(v, vertexInfo)))
                    {
                        return result;
                    }
                }
            }

        // mark current vertex totally visited
        vertexInfo[u->getIndexInMatrix()].color = Colors::Black;

        return result;
    }

public:
    // check whether a graph is acyclic
    optional<pair<N, vector<shared_ptr<Vertex<T>>>*>> isAcyclic() const
    {
        // initialize color and parent for each vertex
        vector<VertexInDFS> verticesInfo(greatest_occupied_index + 1);
        initVerticesForDFS(verticesInfo);

        // initialize end vertices of a cyclic path
        optional<pair<int, int>> cycleStartEnd = nullopt;

        // go through each not visited vertex in a graph
        for (const auto& tmp : vertexList)
        {
            if (verticesInfo[tmp.second->getIndexInMatrix()].color == Colors::White)
            {
                // if algorithm finds cycle - stop search
                if ((cycleStartEnd = dfs_visit(tmp.second, verticesInfo))) break;
            }
        }

        // if algorithm found cycle
        if (cycleStartEnd)
        {
            // reconstruct path
            auto path = constructPath(*cycleStartEnd, verticesInfo);
            // calculate total weight
            int totalWeight = calculateTotalWeight(*path);

            // return these values
            return make_pair(totalWeight, path);
        }
        else return {};
    }

    /* End isAcyclic */

    /* Transpose */
private:
    // simple transposition of a matrix
    void transposeMatrix()
    {
        for (int i = 0; i < greatest_occupied_index + 1; ++i) {
            for (int j = 0; j < i; ++j) {
                auto tmp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tmp;
//                swap(matrix[i][j], matrix[j][i]);
            }
        }
    }

    // swap end vertices of each edge in the edge list
    void swapEndVertices()
    {
        // go through each edge in the edge list
        for (auto& entry : edgeList)
        {
            // obtain an edge
            auto edge = entry.second;

            // swap its end vertices
            auto tmp = edge->getOrigin();
            edge->setOrigin(edge->getDestination());
            edge->setDestination(tmp);
        }
    }
public:

    // change direction of all edges in a graph
    void transpose()
    {
        transposeMatrix();
        swapEndVertices();
    }

    /* End transpose */

public:
    // print info about contents of a graph
    void printInfo() const
    {
        cout << "Matrix (size is " << matrix.size() << "): " << endl;
        for (int i = 0; i < greatest_occupied_index + 1; ++i)
        {
            for (int j = 0; j < greatest_occupied_index + 1; ++j)
            {
                if (matrix[i][j] == nullptr) cout << 0 << ' ';
                else cout << 1 << ' ';
            }
            cout << endl;
        }

        cout << "Vertex List: " << endl;
        for (const auto& tmp : vertexList)
        {
            tmp.second->printInfo();
        }
        cout << endl;

        cout << "Edge List: " << endl;
        for (const auto& tmp : edgeList)
        {
            tmp.second->printInfo();
        }
        cout << endl;
    }

    /* Shortest Path */
private:
    // Describes current state of each vertex with info necessary for Dijkstra's algorithm
    struct VertexInDijkstra
    {
        N distance;
        shared_ptr<Vertex<T>> parent;
        shared_ptr<Vertex<T>> vertex;
    };

    // Initializes info for each vertex
    void initVerticesForDijkstra(vector<VertexInDijkstra>& verticesInfo, shared_ptr<Vertex<T>> source)
    {
        // go through each vertex
        for (const auto& tmp : vertexList)
        {
            auto vertex = tmp.second; // extract pointer to vertex

            // initial distance for source - 0, for other - inf
            if (vertex == source) verticesInfo[vertex->getIndexInMatrix()].distance = 0;
            else verticesInfo[vertex->getIndexInMatrix()].distance = numeric_limits<N>::max();

            // initialize parent of each vertex to null
            verticesInfo[vertex->getIndexInMatrix()].parent = nullptr;
            // specify a pointer for corresponding vertex
            verticesInfo[vertex->getIndexInMatrix()].vertex = vertex;
        }
    }

    // construct the shortest path
    optional<vector<shared_ptr<Vertex<T>>>> constructPathForDijkstra(const shared_ptr<Vertex<T>>& source, const shared_ptr<Vertex<T>>& target, const vector<VertexInDijkstra>& verticesInfo)
    {
        // a vector containing the path
        vector<shared_ptr<Vertex<T>>> path;

        shared_ptr<Vertex<T>> current = target;

        // reconstruct a path
        while (current != source)
        {
            path.push_back(current);
            current = verticesInfo[current->getIndexInMatrix()].parent;
        }

        path.push_back(source);

        // make it in right order
        reverse(path.begin(), path.end());

        return path;
    }

public:
    // find shortest path between two vertices
    optional<vector<shared_ptr<Vertex<T>>>> shortestPath(const shared_ptr<Vertex<T>>& source, const shared_ptr<Vertex<T>>& target, const N& minBandwidth)
    {
        // initialize info about vertices
        vector<VertexInDijkstra> verticesInfo(greatest_occupied_index + 1);
        initVerticesForDijkstra(verticesInfo, source);

        // a 'priority queue' for containing not yet visited vertices
        multiset<pair<N, int>> not_added;
        int sourceIndex = source->getIndexInMatrix();

        not_added.insert({ verticesInfo[sourceIndex].distance, sourceIndex });


        // while there are unvisited vertices
        while (!not_added.empty())
        {
            // extract a vertex with minimal distance
            int indexU = not_added.begin()->second;
            not_added.erase(not_added.begin());

            // if there are any edges from this vertex
            if (auto edges = edgesFrom(verticesInfo[indexU].vertex))
                // go through each neighbour
                for (const auto& edge : *edges)
                {
                    int indexV = edge->getDestination()->getIndexInMatrix();

                    // relaxation process
                    if (verticesInfo[indexV].distance > verticesInfo[indexU].distance + edge->getWeight()
                            && edge->getBandwidth() >= minBandwidth)
                    {
                        not_added.erase({ verticesInfo[indexV].distance, indexV });
                        verticesInfo[indexV].distance = verticesInfo[indexU].distance + edge->getWeight();
                        verticesInfo[indexV].parent = verticesInfo[indexU].vertex;
                        not_added.insert({ verticesInfo[indexV].distance, indexV });
                    }
                }
        }

        // if there is a path between two vertices at all - construct the shortest one
        if (source == target || verticesInfo[target->getIndexInMatrix()].parent != nullptr)
            return constructPathForDijkstra(source, target, verticesInfo);
        else return nullopt; // otherwise, specify that there is none
    }

    /* End shortest path */

private:
    // for enabling hashing pairs
    struct pair_hash
    {
        template <class T1, class T2>
        int operator() (pair<T1, T2> const& v) const
        {
            return std::hash<T1>()(v.first) + std::hash<T2>()(v.second);
        }
    };
private:
    unordered_multimap<pair<shared_ptr<Vertex<T>>, shared_ptr<Vertex<T>>>, shared_ptr<Edge<N, T>>, pair_hash> edgeList;
    unordered_multimap<T, shared_ptr<Vertex<T>>> vertexList;
    vector<vector<shared_ptr<Edge<N, T>>>> matrix;

    // contains indices that can be assigned to a new vertex
    priority_queue<int, vector<int>, greater<>> free_vertices;
    int greatest_occupied_index;

    inline static int initial_matrix_size = 8;
};

// process messages from input
template <typename T, typename N>
void processInput(AdjacencyMatrixGraph<T, N> amg)
{
    string command = " ";
    while (cin >> command)
    {
        if (command == "ADD_VERTEX") {
            string name;
            cin >> name;

            amg.addVertex(name);
        } else if (command == "REMOVE_VERTEX") {
            string name;
            cin >> name;

            auto v1 = amg.findVertex(name);
            amg.removeVertex(*v1);
        } else if (command == "ADD_EDGE") {
            string n1, n2;
            int weight;
            cin >> n1 >> n2 >> weight;

            auto v1 = amg.findVertex(n1);
            auto v2 = amg.findVertex(n2);

            amg.addEdge(*v1, *v2, weight);
        } else if (command == "HAS_EDGE") {
            string n1, n2;
            cin >> n1 >> n2;

            auto v1 = amg.findVertex(n1);
            auto v2 = amg.findVertex(n2);

            if (v1 && v2 && amg.hasEdge(*v1, *v2)) {
                cout << "TRUE" << endl;
            } else {
                cout << "FALSE" << endl;
            }
        } else if (command == "REMOVE_EDGE") {
            string n1, n2;
            cin >> n1 >> n2;

            auto e1 = amg.findEdge(n1, n2);

            amg.removeEdge(*e1);
        } else if (command == "IS_ACYCLIC") {
            auto isAcyclic = amg.isAcyclic();

            if (!isAcyclic)
            {
                cout << "ACYCLIC" << endl;
            }
            else
            {
                int weight = isAcyclic->first;
                cout << weight << " ";
                auto cycle = *isAcyclic->second;
                for (int i = 0; i < cycle.size(); ++i) {
                    cout << cycle[i]->getValue();
                    if (i != cycle.size() - 1) cout << ' ';
                }
                cout << endl;

//                amg.printInfo();
            }
        } else if (command == "TRANSPOSE") {
            amg.transpose();
        }
        else cout << "Unknown command" << endl;

    }
}

// calculate amount of edges in the shortes path
int calcTotalLength(AdjacencyMatrixGraph<int, int>& amg, vector<shared_ptr<Vertex<int>>>& path)
{
    int totalLength = 0;
    for (int i = 0; i < path.size() - 1; ++i)
    {
        auto edge = *amg.findEdge(path[i]->getValue(), path[i + 1]->getValue());
        totalLength += edge->getWeight();
    }

    return totalLength;
}

// calculate the bandwidth of a path
int calcBandwidth(AdjacencyMatrixGraph<int, int> amg, vector<shared_ptr<Vertex<int>>> path)
{
    int bandwidth = numeric_limits<int>::max();
    for (int i = 0; i < path.size() - 1; ++i)
    {
        auto edge = *amg.findEdge(path[i]->getValue(), path[i + 1]->getValue());
        bandwidth = min(edge->getBandwidth(), bandwidth);
    }

    return bandwidth;
}

// a method for managing info about shortest path for the task 'C'
void shortestPathTask()
{
    AdjacencyMatrixGraph<int, int> amg;

    int n1 = 0, n2 = 0;
    int edgeLength = 0, edgeBandwidth = 0;

    int n = 0, m = 0;
    cin >> n >> m;

    for (int i = 0; i < n; ++i)
    {
        amg.addVertex(i + 1);
    }

    for (int i = 0; i < m; ++i)
    {
        cin >> n1 >> n2 >> edgeLength >> edgeBandwidth;


        shared_ptr<Vertex<int>> v1 = *amg.findVertex(n1);
        shared_ptr<Vertex<int>> v2 = *amg.findVertex(n2);

        amg.addEdge(v1, v2, edgeLength, edgeBandwidth);
//        amg.printInfo();
    }

    int source = 0, target = 0;
    int bandwidth = 0;

    cin >> source >> target >> bandwidth;
    auto path = amg.shortestPath(*amg.findVertex(source), *amg.findVertex(target), bandwidth);

    if (!path)
        cout << "IMPOSSIBLE" << endl;
    else
    {
        int nVerticesInPath = path->size();
        int totalLength = calcTotalLength(amg, *path);
        int totalBandwidth = calcBandwidth(amg, *path);
        cout << nVerticesInPath << " " << totalLength << " " << totalBandwidth << endl;

        for (const auto& vertex : *path)
        {
            cout << vertex->getValue() << " ";
        }

        cout << endl;
    }
}

int main() {
    AdjacencyMatrixGraph<string, int> amg;

    processInput(amg);
//    shortestPathTask();
    return 0;
}