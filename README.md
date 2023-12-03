# Parallel Chess Engine
CMU 15618 Project | Casper Wong, Daniela Munoz
## Summary
We are going to parallelize a C chess engine using OpenMP to find the best move for any given board position as fast as possible. We will also analyze the performance and tradeoffs to be made in different parallel implementations.

<details open>
  <summary>Project Milestone</summary>
  
  ## Updated Schedule
  
  | Dates            | Name       | Submit the proposal and research topic                                                                                  |
  |------------------|------------| ------------------------------------------------------------------------------------------------------------------------|
  | 12/4 - 12/6      | Daniela    | Young Brothers Wait Concept Implementation                                                                              |
  | 12/4 - 12/9      | Casper     | Lazy SMP Implementation (Dani may help since this is more involved than Young Brother Wait Concept)                     |
  | 12/10 - 12/12    | Daniela    | Creating graphs to show changes in number of nodes visited and speedup based on number of processors and implementation |
  | 12/10 - 12/12    | Casper     | Estimate ELO of the chess engine or another method (ie. increase in depth parallelized engine can think)                |
  | 12/2 - 12/4      | Both       | Complete final report                                                                                                   |

  ## Milestone report
  At this point, we have completed the sequential and naive parallel implementations of our chess engine. After creating the sequential version and before implementing the naive parallelism, we created a profiling script that would allow us to ensure that the engine is in fact picking the best move. We do this by taking 16 different chess boards from various openings, middlegames, and endgames, and ensuring that the move the engine picks for that board is optimal. We also track the time it takes to pick a move and how many nodes are visited. This allows us to better understand the tradeoffs of our parallel programs.

For the naive parallel implementation, we simply parallelize the outermost loop in the function that picks the best move. The workers share the work of looping through every possible move to evaluate a score based on the mini-max function going through future states of that board. We do not parallelize the mini-max function in further depth yet. 

So far, we have seen the following improvements (time table on the left, speedup table on the right):
  <img width="382" alt="Screenshot 2023-12-03 at 4 06 32 PM" src="https://github.com/CasperWong-jpg/ParallelChess/assets/58316207/9dbe51fc-b3ee-410c-9be6-47d54a628d26">
<img width="387" alt="Screenshot 2023-12-03 at 4 06 56 PM" src="https://github.com/CasperWong-jpg/ParallelChess/assets/58316207/77e813c0-25e4-453b-8eaf-80a94b0fd404">


  Average speedups
  1core(s)    1.000000
  2core(s)    1.882944
  4core(s)    3.151937
  8core(s)    4.263526
  
  We believe that we are well on track to complete the deliverables stated in our proposal. We are already seeing an average of a little over 2x speedup with 4 processors on the naive implementation, which is on par with others’ work and was part of our “hope to achieve” section. However, we can see that the speedup does not scale well as we further increase core count to 8 and more cores, which we hope to focus on.
  
  We plan on completing 1-2 more involved parallelized implementations of our engine to see if we can increase the speedup at all and better understand some trade offs of different methods. Specifically, we are looking to implement Young Brother’s Wait Concept and Lazy SMP, but there is a chance that we will not successfully complete both since they are rather involved.
  
  Also, we have not yet found the ELO of our implementation, but still hope to estimate this by putting our engine against others with differing ratings and documenting how often it wins. 
  
  During the poster session, we will have a demo that shows our engine playing against others of different ratings and allow people to play against our engine as well. We may also include graphs if we find interesting results between the different parallel versions or by using different numbers of processors.


  
</details>

<details>
  <summary>Project Proposal</summary>
  ## Background
  It is estimated there are between 10^111 and 10^123 chess board positions, including illegal moves. Without illegal moves, this number drops to 10^40. This is still quite a large number, so it is not possible for a chess engine to analyze all possible positions. For example, Stockfish 8, one of the most advanced chess engines, is only able to think up to 22 moves ahead in 1 second, and it takes exponentially longer to think another move ahead. Given that a typical chess game is 10 or 30 minutes, we cannot afford to spend more than seconds to think about each move.
  
  Typically, chess engines are implemented using Minimax, a tree algorithm where new board positions are generated by making different moves. We can generate positions up to a certain depth of moves and use different heuristics to give each move an estimated score to then choose the best one. 
  
  Generating all possible moves and determining which is the best one takes a long time sequentially. So, there is a lot of potential for parallelism as the simulations for each move are independent of each other. 
  
  ## The Challenge
  Alpha-beta pruning is commonly used to make Minimax significantly faster, where large portions of the game tree can be avoided if a better move has already been found. 
  
  We plan on using alpha-beta pruning to optimize this algorithm, however this introduces a lot of workload imbalance. Each time a branch is cut and the algorithm decides to no longer consider any of its descendents, that process will idle while waiting for the other threads to complete creating their game tree. So, we need to find a way to balance and share workloads such that threads do not idle unnecessarily. 
  
  Another challenge we see ourselves facing is the communication between threads. Each turn, the Chess engine is considering over a million possible moves (thinking 5 moves ahead). We need to ensure that we do not perform redundant computations between threads, and that each thread is able to communicate with one another about the best move they can achieve. 
  
  ## Resources
  One of the group partners has previously built a chess engine in Python and is attempting to port it to C. If we are unable to do so while staying on schedule, we plan on using an existing chess engine built in C, such as TSCP or Ethereal. 
  
  We will be using this [Master’s paper](https://www.duo.uio.no/bitstream/handle/10852/53769/master.pdf) as a reference and benchmark for our own program. 
  
  We also plan on using the Lichess API to create the live demo and get ELO rankings for our versions. 
  
  ## Goals and Deliverables
  ### Plan to Achieve
  In the research paper that we studied, Østensen was been able to achieve about a 2x speedup with 4 processors, so we also plan to achieve at least a 2x speedup with up to 8 processors. We hope that this enables our Chess Engine to think one further move ahead without a significant time increase.
  
  We also plan on profiling and generating graphs with useful information about the tradeoffs between the number of processors, implementation method (ie. scheduling type), and speedup. 
  
  ### Hope to Achieve
  We hope to achieve a 2x speedup or greater on 4 processors to be on par with others’ work. We also hope to increase the ELO by 100-200 from the sequential to the parallel version.
  
  ### Showcase
  We plan on developing a live demo by hooking it up to the Lichess bot API for people to play against! This can also be used to evaluate the final ELO of the chess engine. 
  
  ## Platform Choice
  We will be using C as our programming language since many of the leading Chess engines are created in this language. We plan on using OpenMP because it allows us to split up work among cores easily using the fork-join model. 
  
  Finally, we will be running this program on regular laptops, such as Macbook Pro with 8 cores or the GHC computers, since we would like to host the Chess Engine locally. 
  
  ## Schedule
  | Week 1 11/12 - 11/18 | Submit the proposal and research topic                                                                       |
  |----------------------|--------------------------------------------------------------------------------------------------------------|
  | Week 2 11/19 - 11/25 | Build sequential implementation Perform timing and performance profiling                                     |
  | Week 3 11/26 - 12/2  | Develop a naive parallel implementation, Complete and submit milestone report                                |
  | Week 4 12/3 - 12/9   | Optimize parallel implementations – explore various OpenMP methods  Perform timing and performance profiling |
  | Week 5 12/10 - 12/14 | Complete final parallel implementation, Complete and submit the final report                                 |
</details>

## How to run
To run Lichess API:

Change `lichess_bot/config.yml` OAuth token to bot account you own\
`make lichess` \
`cd lichess_bot` \
`python3 lichess-bot.py`
