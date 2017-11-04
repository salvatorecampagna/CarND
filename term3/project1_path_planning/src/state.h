#ifndef STATE_H
#define STATE_H

// States for the Finite State Machine:
// START: start state of the car
// KEEP_LANE_STATE: stay in the current lane
// CHANGE_LEFT_STATE: change lane going to the left lane
// CHANGE_RIGHT_STATE: change lane going to the right

enum class STATE
{
  START_STATE,
  KEEP_LANE_STATE,
  CHANGE_LEFT_STATE,
  CHANGE_RIGHT_STATE
};

#endif
