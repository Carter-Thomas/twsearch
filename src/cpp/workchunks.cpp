#include "workchunks.h"
#include "canon.h"
#include "rotations.h"
#include "solve.h"
#include "threads.h"
#include <iostream>
#include <openacc.h>

int randomstart;
static vector<allocsetval> seen;
static int lastsize;

// GPU-accelerated work chunk generation
vector<ull> makeworkchunks(const puzdef &pd, int d, setval symmreduce,
                           int microthreadcount) {
  vector<int> workstates;
  vector<ull> workchunks;
  workchunks.push_back(1);
  workstates.push_back(0);
  
  if (d >= 3) {
    if (pd.totsize != lastsize) {
      lastsize = pd.totsize;
      seen.clear();
    }
    
    stacksetval p1(pd), p2(pd), p3(pd);
    int nmoves = pd.moves.size();
    int chunkmoves = 0;
    ull mul = 1;
    int mythreads = microthreadcount * numthreads;
    ll hashmod = 100 * mythreads;
    vector<int> hashfront(hashmod, -1);
    vector<int> hashprev;
    int seensize = 0;
    
    // Initialize GPU
    #pragma acc init
    
    while (chunkmoves + 3 < d && (int)workchunks.size() < 40 * mythreads) {
      vector<ull> wc2;
      vector<int> ws2;
      
      if (pd.rotgroup.size() > 1) {
        // GPU-accelerated rotation group processing
        #pragma acc data copyin(pd, workchunks[0:workchunks.size()], workstates[0:workstates.size()])
        {
          for (int i = 0; i < (int)workchunks.size(); i++) {
            ull pmv = workchunks[i];
            int st = workstates[i];
            ull mask = canonmask[st];
            ull t = pmv;
            pd.assignpos(p1, symmreduce);
            
            // Apply initial moves
            while (t > 1) {
              domove(pd, p1, t % nmoves);
              t /= nmoves;
            }
            
            // GPU parallel move generation
            #pragma acc parallel loop present(pd)
            for (int mv = 0; mv < nmoves; mv++) {
              if (quarter && pd.moves[mv].cost > 1)
                continue;
              if ((mask >> pd.moves[mv].cs) & 1)
                continue;
              
              pd.mul(p1, pd.moves[mv].pos, p2);
              if (!pd.legalstate(p2))
                continue;
              
              slowmodm2(pd, p2, p3);
              int h = fasthash(pd.totsize, p3) % hashmod;
              int isnew = 1;
              
              #pragma acc atomic capture
              {
                for (int j = hashfront[h]; j >= 0; j = hashprev[j]) {
                  if (pd.comparepos(p3, seen[j]) == 0) {
                    isnew = 0;
                    break;
                  }
                }
              }
              
              if (isnew) {
                #pragma acc critical
                {
                  wc2.push_back(pmv + (nmoves + mv - 1) * mul);
                  ws2.push_back(canonnext[st][pd.moves[mv].cs]);
                  
                  if (seensize < (int)seen.size()) {
                    pd.assignpos(seen[seensize], p3);
                  } else {
                    seen.push_back(allocsetval(pd, p3));
                  }
                  
                  hashprev.push_back(hashfront[h]);
                  hashfront[h] = seensize;
                  seensize++;
                }
              }
            }
          }
        }
      } else {
        // Non-rotation group processing with GPU acceleration
        #pragma acc data copyin(pd, workchunks[0:workchunks.size()], workstates[0:workstates.size()])
        {
          #pragma acc parallel loop
          for (int i = 0; i < (int)workchunks.size(); i++) {
            ull pmv = workchunks[i];
            int st = workstates[i];
            ull mask = canonmask[st];
            const vector<int> &ns = canonnext[st];
            
            #pragma acc loop seq
            for (int mv = 0; mv < nmoves; mv++) {
              if (0 == ((mask >> pd.moves[mv].cs) & 1)) {
                #pragma acc critical
                {
                  wc2.push_back(pmv + (nmoves + mv - 1) * mul);
                  ws2.push_back(ns[pd.moves[mv].cs]);
                }
              }
            }
          }
        }
      }
      
      swap(wc2, workchunks);
      swap(ws2, workstates);
      chunkmoves++;
      mul *= nmoves;
      
      if (mul >= (1ULL << 62) / nmoves) {
        cout << "Mul got too big " << nmoves << " " << mul << endl;
        break;
      }
    }
    
    // Randomize if needed
    if (randomstart) {
      #pragma acc parallel loop
      for (int i = 0; i < (int)workchunks.size(); i++) {
        int j = i + myrand(workchunks.size() - i);
        #pragma acc atomic capture
        {
          swap(workchunks[i], workchunks[j]);
          swap(workstates[i], workstates[j]);
        }
      }
    }
    
    #pragma acc shutdown
  }
  
  return workchunks;
}
