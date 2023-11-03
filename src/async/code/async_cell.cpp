#include <thread>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <future>
#include <vector>
#include <sstream>
#include <functional>


// A class representing the state of execution for a 1-dimensional
// cellular automaton
class AsyncCellSim{

    // The 8-bit value representing the type of 1D automaton
    unsigned char   rule;

    // The width of the "world" and how many generations that world
    // should be simulated for
    size_t width;
    size_t generation_count;

    // A "dummy" future used to get the state of non-existant cells
    std::shared_future<bool> edge;

    // Points to a flattened 2D array holding futures for the state
    // of every cell at every time step
    std::shared_future<bool> *cell_states;

    // Uses ANSI escapes to update a specific cell's displayed state
    void display_cell(bool state, size_t position, size_t generation) {
        // The symbol for a cell
        char symbol = state ? '#' : ' ';
        // A stringstream to make formatting easier
        std::stringstream ss;
        // The amount the cursor needs to move vertically and
        // horizontally to mark the correct spot
        size_t y = generation_count - generation;
        size_t x = position;
        // Move into position
        if(y>0){
            ss << "\033[" << y << "A";
        }
        if(x>0){
            ss << "\033[" << x << "C";
        }
        // Mark with updated symbol
        ss << symbol;
        // Reset cursor location
        if(y>0){
            ss << "\033[" << y << "B";
        }
        ss << "\033[" << (x+1) << "D";
        // Convert stream to string and write directly to stdout
        // The lack of buffering means different thread's outputs
        // will not interleave.
        std::string full_sequence = ss.str();
        std::cout.write(full_sequence.c_str(),full_sequence.length());
    }

    // Simply returns the input state. Used to represent the initial
    // state of each cell
    bool preset_cell(bool state) {
        return state;
    }

    // Calculates the value of the cell at position `pos` at generation `gen`, using
    // the futures representing the previous state of this cell (at generation `gen`-1)
    // and the cells adjacent to it.
    bool async_cell( size_t pos, size_t gen ) {


        // If this is the leftmost cell, use the edge future in place of the non-existent
        // prior left cell state
        std::shared_future<bool> & left = (pos==0) ? edge : cell_states[(gen-1)*width+pos-1];

        // There should always be a previous state for any cell at a non-zero timestep
        std::shared_future<bool> & middle = cell_states[(gen-1)*width+pos];

        // If this is the rightmost cell, use the edge future in place of the non-existent
        // prior right cell state
        std::shared_future<bool> & right = (pos==(width-1)) ? edge : cell_states[(gen-1)*width+pos+1];

        // Calculate the index of the bit that stores the next state of the cell
        unsigned char index = right.get() | (middle.get() << 1) | (left.get() << 2);

        // A dummy wait, used to simulate the latency of heavy processing
        std::this_thread::sleep_for(std::chrono::milliseconds(rand()%50));

        // Find the next state
        bool result = (rule >> index) & 1;

        // Update the display
        display_cell(result,pos,gen);

        return result;
    }

    // Prints a blank canvas for the cell update to overwrite
    void blank_canvas(std::vector<bool> starting_state) {
        // The canvas is just a generation_count-by-width
        // rectangle of '`' characters
        for (size_t gen=0; gen<generation_count; gen++) {
            for (size_t pos=0; pos<width; pos++) {
                char symbol = '`';
                if(gen == 0){
                    symbol = starting_state[pos] ? '#' : ' ';
                }
                std::cout << symbol;
            }
            std::cout << '\n';
        }
    }


    public:

    // Constructor.
    AsyncCellSim(
        size_t generation_count,
        std::vector<bool> starting_state,
        unsigned char rule, std::launch launch_mode
    )
        : width(starting_state.size())
        , generation_count(generation_count)
        , rule(rule)
    {
        blank_canvas(starting_state);

        // Set up a future to query for cells that are beyond the bounds of the simulation
        edge = std::async(
            std::bind(&AsyncCellSim::preset_cell,this,false)
        );

        // Allocate flattened 2D array of futures, one for each position/generation
        cell_states = new std::shared_future<bool>[width*generation_count];

        // Initialize the grid of futures
        for(int gen=0; gen<generation_count; gen++){
            for(int pos=0; pos<width; pos++){
                if(gen == 0){
                    // If this is the first generation, use the preset_cell function to pass through
                    // the preset value
                    cell_states[gen*width+pos] = std::async(
                        std::bind(&AsyncCellSim::preset_cell,this,starting_state[pos])
                    );
                } else {
                    // Dispatch the calculation for that cell state
                    cell_states[gen*width+pos] = std::async(
                        launch_mode,
                        std::bind(&AsyncCellSim::async_cell,this,pos,gen)
                    );
                }
            }
        }

    }

    // Destruction only requires the deletion of the futures grid
    ~AsyncCellSim() {
        delete[] cell_states;
    }

    // A way to wait for specific cell states
    bool query(size_t position, size_t generation) {
        return cell_states[generation*width+position].get();
    }

};


int main(int argc, char *argv[]) {

    srand(time(0));
    unsigned char const rule = 18;

    std::vector<bool> starting_state;
    size_t const width = 64;
    size_t const gen_count = 32;
    starting_state.resize(width);

    // Set only the middlemost cell as "alive"
    for(int pos=0; pos<width; pos++){
        starting_state[pos] = (pos == (width/2));
    }

    // Set up the simulation
    AsyncCellSim sim(
        gen_count,
        starting_state,
        rule,
        std::launch::async | std::launch::deferred
    );

    // Request the final states of all cells
    for(size_t i=0; i<width; i++){
        sim.query(i,gen_count-1);
    }

}

