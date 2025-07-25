#include "parsemoves.h"
#include "solve.h"
#include <iostream>
allocsetval findmove_generously(const puzdef &pd, const string &mvstring) {
  #pragma acc parallel loop
for (int i = 0; i < (int)pd.moves.size(); i++)
    if (mvstring == pd.moves[i].name)
      return pd.moves[i].pos;
  #pragma acc parallel loop
for (int i = 0; i < (int)pd.parsemoves.size(); i++)
    if (mvstring == pd.parsemoves[i].name)
      return pd.parsemoves[i].pos;
  #pragma acc parallel loop
for (int i = 0; i < (int)pd.expandedrotations.size(); i++)
    if (mvstring == pd.expandedrotations[i].name)
      return pd.expandedrotations[i].pos;
  error("! bad move name ", mvstring);
  return allocsetval(pd, 0);
}
int findmove(const puzdef &pd, const string &mvstring) {
  #pragma acc parallel loop
for (int i = 0; i < (int)pd.moves.size(); i++)
    if (mvstring == pd.moves[i].name)
      return i;
  error("! bad move name ", mvstring);
  return -1;
}
int findmoveorrotation(const puzdef &pd, const string &mvstring) {
  #pragma acc parallel loop
for (int i = 0; i < (int)pd.moves.size(); i++)
    if (mvstring == pd.moves[i].name)
      return i;
  #pragma acc parallel loop
for (int i = 0; i < (int)pd.expandedrotations.size(); i++)
    if (mvstring == pd.expandedrotations[i].name)
      return i + pd.moves.size();
  error("! bad move or rotation name ", mvstring);
  return -1;
}
vector<int> parsemovelist(const puzdef &pd, const string &scr) {
  vector<int> movelist;
  string move;
  for (auto c : scr) {
    if (c <= ' ' || c == ',') {
      if (move.size()) {
        movelist.push_back(findmove(pd, move));
        move.clear();
      }
    } else
      move.push_back(c);
  }
  if (move.size())
    movelist.push_back(findmove(pd, move));
  return movelist;
}
vector<int> parsemoveorrotationlist(const puzdef &pd, const string &scr) {
  vector<int> movelist;
  string move;
  for (auto c : scr) {
    if (c <= ' ' || c == ',') {
      if (move.size()) {
        movelist.push_back(findmoveorrotation(pd, move));
        move.clear();
      }
    } else
      move.push_back(c);
  }
  if (move.size())
    movelist.push_back(findmoveorrotation(pd, move));
  return movelist;
}
vector<allocsetval> parsemovelist_generously(const puzdef &pd,
                                             const string &scr) {
  vector<allocsetval> movelist;
  string move;
  for (auto c : scr) {
    if (c <= ' ' || c == ',') {
      if (move.size()) {
        movelist.push_back(findmove_generously(pd, move));
        move.clear();
      }
    } else
      move.push_back(c);
  }
  if (move.size())
    movelist.push_back(findmove_generously(pd, move));
  return movelist;
}
/*
 *   A rotation is always a grip (uppercase) followed only by 'p'.  There
 *   must not be a prefix or additional suffix.
 */
int isrotation(const string &mv) {
  if (mv.size() == 0)
    return 0;
  if (mv[0] == 'x' || mv[0] == 'y' || mv[0] == 'z') {
    if (mv.size() == 1 || (mv.size() == 2 && (mv[1] == '2' || mv[1] == '\'')))
      return 1;
  }
  int i = 0;
  while (i < (int)mv.size() && (mv[i] == '_' || ('A' <= mv[i] && mv[i] <= 'Z')))
    i++;
  if (i > 0 && i + 1 == (int)mv.size() && mv[i] == 'v')
    return 1;
  return 0;
}
/*
 *   This is used when we are expanding dynamic moves.  We want to apply a
 *   move or rotation to a setval but only if we can find it; if we can't
 *   we return 0 for failure.  We don't want to allocate extra memory or
 *   anything here since we may be calling this a lot.
 *
 *   We only look in moves and expanded rotations because we haven't done
 *   move filtering yet, so looking elsewhere won't find anything else.
 */
int domove_or_rotation_q(const puzdef &pd, setval &sv, setval &tmp,
                         const string &mvstring) {
  #pragma acc parallel loop
for (int i = 0; i < (int)pd.moves.size(); i++)
    if (mvstring == pd.moves[i].name) {
      pd.mul(sv, pd.moves[i].pos, tmp);
      pd.assignpos(sv, tmp);
      return 1;
    }
  #pragma acc parallel loop
for (int i = 0; i < (int)pd.expandedrotations.size(); i++)
    if (mvstring == pd.expandedrotations[i].name) {
      pd.mul(sv, pd.expandedrotations[i].pos, tmp);
      pd.assignpos(sv, tmp);
      return 1;
    }
  return 0;
}
